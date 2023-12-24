#main.py主程序

import glob
import os
import numpy
import pandas
import torch
import random

import utils
from server import Server
from client import Client

#程序开始运行
print("Program start...")
#获得命令行实例，直接args.参数名调用
conf = utils.get_conf()
#获得日志实例
logging = utils.get_logger(__name__)
#如果设定的gpu序号不合理，报错
assert conf["num_gpus"] <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"
#获得device实例
device = utils.get_device(gpus=conf["num_gpus"])

#用于快速跳过
if conf["flag"] == False:
    #数据准备
    print("Start preparing data...")
    #文件实例化
    transaction_df = pandas.read_csv('./data/input/train_transaction.csv')
    identity_df = pandas.read_csv('./data/input/train_identity.csv')
    test_transaction = pandas.read_csv('./data/input/test_transaction.csv')
    test_identity = pandas.read_csv('./data/input/test_identity.csv')
    #设置表头列表
    id_cols = ['card1','card2','card3','card4','card5','card6','ProductCD','addr1','addr2','P_emaildomain','R_emaildomain']
    cat_cols = ['M1','M2','M3','M4','M5','M6','M7','M8','M9']
    #指定训练集比例
    train_data_ratio = 0.8
    #训练集长度，测试集ID构成列表
    n_train = int(transaction_df.shape[0]*train_data_ratio)
    test_ids = transaction_df.TransactionID.values[n_train:]
    get_fraud_frac = lambda series: 100 * sum(series)/len(series)
    #显示各个数据集诈骗率
    print("Percent fraud for train transactions: {}".format(get_fraud_frac(transaction_df.isFraud[:n_train])))
    print("Percent fraud for test transactions: {}".format(get_fraud_frac(transaction_df.isFraud[n_train:])))
    print("Percent fraud for all transactions: {}".format(get_fraud_frac(transaction_df.isFraud)))
    #将测试集的ID写入文件
    with open('./data/working/test.csv', 'w') as f:
        f.writelines(map(lambda x: str(x) + "\n", test_ids))
    #非特征列表，前面有具体名字的列
    non_feature_cols = ['isFraud', 'TransactionDT'] + id_cols
    #特征列表，后面没具体名字的列
    feature_cols = [col for col in transaction_df.columns if col not in non_feature_cols]
    #给特征列表设置独热编码，生成词向量
    features = pandas.get_dummies(transaction_df[feature_cols], columns=cat_cols).fillna(0) #填充空值
    features['TransactionAmt'] = features['TransactionAmt'].apply(numpy.log10) #对金额取对数
    #将编码后特征存入文件
    features.to_csv('./data/working/features.csv', index=False, header=False)
    #将这两个列存入文件
    transaction_df[['TransactionID', 'isFraud']].to_csv('./data/working/tags.csv', index=False)
    #边类型，一部分transaction，一部分identity的列名编程列表
    edge_types = id_cols + list(identity_df.columns)
    #把两个文件拼起来
    all_id_cols = ['TransactionID'] + id_cols
    full_identity_df = transaction_df[all_id_cols].merge(identity_df, on='TransactionID', how='left')
    #遍历文件，得到边列表，并存入文件
    edges = {}
    for etype in edge_types:
        edgelist = full_identity_df[['TransactionID', etype]].dropna()
        edgelist.to_csv('./data/working/relation_{}_edgelist.csv'.format(etype), index=False, header=True)
        edges[etype] = edgelist
    #打印结束提示
    print("...Data preparation finished")


#操作进行
print("Start operating...")
#操作部分
file_list = glob.glob('./data/working/*edgelist.csv')
edges = ",".join(map(lambda x: x.split("/")[-1], [file for file in file_list if "relation" in file]))
conf["edges"] = edges
conf["edges"] = utils.get_edgelists('relation*', conf["training_dir"])
g, features, target_id_to_node, id_to_node = utils.construct_graph(conf["training_dir"],
                                                                conf["edges"],
                                                                conf["nodes"],
                                                                conf["target_ntype"])
mean, stdev, features = utils.normalize(torch.from_numpy(features))
g.nodes['target'].data['features'] = features

print("Getting labels...")
n_nodes = g.number_of_nodes('target')
labels, _, test_mask = utils.get_labels(target_id_to_node,
                                            n_nodes,
                                            conf["target_ntype"],
                                            os.path.join(conf["training_dir"], conf["labels"]),
                                            os.path.join(conf["training_dir"], conf["new_accounts"]) )
print("...Got labels")

labels = torch.from_numpy(labels).float()
test_mask = torch.from_numpy(test_mask).float()
n_nodes = torch.sum(torch.tensor([g.number_of_nodes(n_type) for n_type in g.ntypes]))
n_edges = torch.sum(torch.tensor([g.number_of_edges(e_type) for e_type in g.etypes]))
print("...Operating finished")


print("Initializing Model...")
in_feats = features.shape[1]
n_classes = 2
ntype_dict = {n_type: g.number_of_nodes(n_type) for n_type in g.ntypes}
model = utils.get_model(ntype_dict, g.etypes, conf, in_feats, n_classes, device)
print("...Model Initialized")

print("Starting Model training...")
features = features.to(device)
labels = labels.long().to(device)
test_mask = test_mask.to(device)
g.to(device)
loss = torch.nn.CrossEntropyLoss()
utils.initial_record()

#创建实例
server = Server(conf, model)
clients = []
for c in range(conf["number_clients"]):
    clients.append(Client(conf, model, c))

for e in range(conf["global_epochs"]):
		print("Global epoch {:03d}".format(e))
		#k是每次选取多少个客户端
		candidates = random.sample(clients, conf["k"])
		weight_accumulator = {}
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)
		for c in candidates:
            #这里传入的参数需要注意
			diff = c.local_train(model, loss, features, labels, g, g,
                test_mask, device, conf["local_epochs"], conf["threshold"],  conf["compute_metrics"])
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])
		#更新全局模型
		server.model_aggregate(weight_accumulator)

#最后输出结果
class_preds, pred_proba = utils.get_model_class_predictions(model,
													g,
													features,
													labels,
													device,
													threshold=conf["threshold"])
if conf["compute_metrics"]:
	acc, f1, p, r, roc, pr, ap, cm = utils.get_metrics(class_preds, pred_proba, labels.numpy(), test_mask.numpy(), './data/working')
	print("Metrics")
	print("""Confusion Matrix:
							{}
							f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, acc: {:.4f}, roc: {:.4f}, pr: {:.4f}, ap: {:.4f}
						""".format(cm, f1, p, r, acc, roc, pr, ap))

print("...Model training ended")
print("...Program ended successfully!")
