#util.py工具模块

import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import dgl
import argparse
import logging
import torch
import re
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy
import pandas
import pickle
import torch as th
import numpy as np
import pandas as pd

from models import HeteroRGCN

#命令行解析
def get_conf():
    parser = argparse.ArgumentParser(description="Fraud Detection")
    #用conf指定json文件
    parser.add_argument('-c', '--conf', dest='conf', default="./conf.json")
    parser.parse_args()
    args = parser.parse_args()
    with open(args.conf, 'r') as f:
        conf = json.load(f)	
    return conf

#设置日志格式
def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger

#设置自动设置GPU加速
def get_device(no_cuda=False, gpus='0'):
    return torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")

###############
#以下是图处理函数
#补充一个处理错误的函数
def string_to_float(string):
    try:
        return float(string)
    except:
        return 0.0

def get_features(id_to_node, node_features):
    indices, features, new_nodes = [], [], []
    max_node = max(id_to_node.values())
    with open(node_features, "r") as fh:
        for line in fh:
            node_feats = line.strip().split(",")
            node_id = node_feats[0]
            feats = np.array(list(map(string_to_float, node_feats[1:])))
            features.append(feats)
            if node_id not in id_to_node:
                max_node += 1
                id_to_node[node_id] = max_node
                new_nodes.append(max_node)
            indices.append(id_to_node[node_id])
    features = numpy.array(features).astype('float32')
    features = features[numpy.argsort(indices), :]
    return features, new_nodes

def get_labels(id_to_node, n_nodes, target_node_type, labels_path, masked_nodes_path, additional_mask_rate=0):
    node_to_id = {v: k for k, v in id_to_node.items()}
    user_to_label = pandas.read_csv(labels_path).set_index(target_node_type)
    labels = user_to_label.loc[map(int, pd.Series(node_to_id)[np.arange(n_nodes)].values)].values.flatten()
    masked_nodes = read_masked_nodes(masked_nodes_path)
    train_mask, test_mask = _get_mask(id_to_node, node_to_id, n_nodes, masked_nodes,
                                      additional_mask_rate=additional_mask_rate)
    return labels, train_mask, test_mask

def read_masked_nodes(masked_nodes_path):
    with open(masked_nodes_path, "r") as fh:
        masked_nodes = [line.strip() for line in fh]
    return masked_nodes

def _get_mask(id_to_node, node_to_id, num_nodes, masked_nodes, additional_mask_rate):
    train_mask = np.ones(num_nodes)
    test_mask = np.zeros(num_nodes)
    for node_id in masked_nodes:
        train_mask[id_to_node[node_id]] = 0
        test_mask[id_to_node[node_id]] = 1
    if additional_mask_rate and additional_mask_rate < 1:
        unmasked = np.array([idx for idx in range(num_nodes) if node_to_id[idx] not in masked_nodes])
        yet_unmasked = np.random.permutation(unmasked)[:int(additional_mask_rate*num_nodes)]
        train_mask[yet_unmasked] = 0
    return train_mask, test_mask

def _get_node_idx(id_to_node, node_type, node_id, ptr):
    if node_type in id_to_node:
        if node_id in id_to_node[node_type]:
            node_idx = id_to_node[node_type][node_id]
        else:
            id_to_node[node_type][node_id] = ptr
            node_idx = ptr
            ptr += 1
    else:
        id_to_node[node_type] = {}
        id_to_node[node_type][node_id] = ptr
        node_idx = ptr
        ptr += 1
    return node_idx, id_to_node, ptr

def parse_edgelist(edges, id_to_node, header=False, source_type='user', sink_type='user'):
    edge_list = []
    rev_edge_list = []
    source_pointer, sink_pointer = 0, 0
    with open(edges, "r") as fh:
        for i, line in enumerate(fh):
            source, sink = line.strip().split(",")
            if i == 0:
                if header:
                    source_type, sink_type = source, sink
                if source_type in id_to_node:
                    source_pointer = max(id_to_node[source_type].values()) + 1
                if sink_type in id_to_node:
                    sink_pointer = max(id_to_node[sink_type].values()) + 1
                continue
            source_node, id_to_node, source_pointer = _get_node_idx(id_to_node, source_type, source, source_pointer)
            if source_type == sink_type:
                sink_node, id_to_node, source_pointer = _get_node_idx(id_to_node, sink_type, sink, source_pointer)
            else:
                sink_node, id_to_node, sink_pointer = _get_node_idx(id_to_node, sink_type, sink, sink_pointer)
            edge_list.append((source_node, sink_node))
            rev_edge_list.append((sink_node, source_node))
    return edge_list, rev_edge_list, id_to_node, source_type, sink_type

def read_edges(edges, nodes=None):
    node_pointer = 0
    id_to_node = {}
    features = []
    sources, sinks = [], []
    if nodes is not None:
        with open(nodes, "r") as fh:
            for line in fh:
                node_feats = line.strip().split(",")
                node_id = node_feats[0]
                if node_id not in id_to_node:
                    id_to_node[node_id] = node_pointer
                    node_pointer += 1
                    if len(node_feats) > 1:
                        feats = np.array(list(map(float, node_feats[1:])))
                        features.append(feats)
        with open(edges, "r") as fh:
            for line in fh:
                source, sink = line.strip().split(",")
                sources.append(id_to_node[source])
                sinks.append(id_to_node[sink])
    else:
        with open(edges, "r") as fh:
            for line in fh:
                source, sink = line.strip().split(",")
                if source not in id_to_node:
                    id_to_node[source] = node_pointer
                    node_pointer += 1
                if sink not in id_to_node:
                    id_to_node[sink] = node_pointer
                    node_pointer += 1
                sources.append(id_to_node[source])
                sinks.append(id_to_node[sink])
    return sources, sinks, features, id_to_node

def get_edgelists(edgelist_expression, directory):
    if "," in edgelist_expression:
        return edgelist_expression.split(",")
    files = os.listdir(directory)
    compiled_expression = re.compile(edgelist_expression)
    return [filename for filename in files if compiled_expression.match(filename)]

def construct_graph(training_dir, edges, nodes, target_node_type):
    print("Getting relation graphs from the following edge lists : {} ".format(edges))
    edgelists, id_to_node = {}, {}
    for i, edge in enumerate(edges):
        edgelist, rev_edgelist, id_to_node, src, dst = parse_edgelist(os.path.join(training_dir, edge), id_to_node, header=True)
        if src == target_node_type:
            src = 'target'
        if dst == target_node_type:
            dst = 'target'
        if src == 'target' and dst == 'target':
            print("Will add self loop for target later......")
        else:
            edgelists[(src, src + '<>' + dst, dst)] = edgelist
            edgelists[(dst, dst + '<>' + src, src)] = rev_edgelist
            print("Read edges for {} from edgelist: {}".format(src + '<' + dst + '>', os.path.join(training_dir, edge)))
    # get features for target nodes
    features, new_nodes = get_features(id_to_node[target_node_type], os.path.join(training_dir, nodes))
    print("Read in features for target nodes")
    # add self relation
    edgelists[('target', 'self_relation', 'target')] = [(t, t) for t in id_to_node[target_node_type].values()]
    g = dgl.heterograph(edgelists)
    print(
        "Constructed heterograph with the following metagraph structure: Node types {}, Edge types{}".format(
            g.ntypes, g.canonical_etypes))
    print("Number of nodes of type target : {}".format(g.number_of_nodes('target')))
    g.nodes['target'].data['features'] = th.from_numpy(features)
    target_id_to_node = id_to_node[target_node_type]
    id_to_node['target'] = target_id_to_node
    del id_to_node[target_node_type]
    return g, features, target_id_to_node, id_to_node

def get_metrics(pred, pred_proba, labels, mask, out_dir):
    labels, mask = labels, mask
    labels, pred, pred_proba = labels[numpy.where(mask)], pred[numpy.where(mask)], pred_proba[numpy.where(mask)]
    acc = ((pred == labels)).sum() / mask.sum()
    true_pos = (numpy.where(pred == 1, 1, 0) + numpy.where(labels == 1, 1, 0) > 1).sum()
    false_pos = (numpy.where(pred == 1, 1, 0) + numpy.where(labels == 0, 1, 0) > 1).sum()
    false_neg = (numpy.where(pred == 0, 1, 0) + numpy.where(labels == 1, 1, 0) > 1).sum()
    true_neg = (numpy.where(pred == 0, 1, 0) + numpy.where(labels == 0, 1, 0) > 1).sum()
    precision = true_pos/(true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos/(true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2*(precision*recall)/(precision + recall) if (precision + recall) > 0 else 0
    confusion_matrix = numpy.DataFrame(numpy.array([[true_pos, false_pos], [false_neg, true_neg]]),
                                    columns=["labels positive", "labels negative"],
                                    index=["predicted positive", "predicted negative"])
    ap = average_precision_score(labels, pred_proba)
    fpr, tpr, _ = roc_curve(labels, pred_proba)
    prc, rec, _ = precision_recall_curve(labels, pred_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(rec, prc)
    save_roc_curve(fpr, tpr, roc_auc, os.path.join(out_dir, "roc_curve.png"))
    save_pr_curve(prc, rec, pr_auc, ap, os.path.join(out_dir, "pr_curve.png"))
    return acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix

def save_roc_curve(fpr, tpr, roc_auc, location):
    f = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model ROC curve')
    plt.legend(loc="lower right")
    f.savefig(location)
    
def save_pr_curve(fpr, tpr, pr_auc, ap, location):
    f = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Model PR curve: AP={0:0.2f}'.format(ap))
    plt.legend(loc="lower right")
    f.savefig(location)
    
def save_graph_drawing(g, location):
    plt.figure(figsize=(12, 8))
    node_colors = {node: 0.0 if 'user' in node else 0.5 for node in g.nodes()}
    nx.draw(g, node_size=10000, pos=nx.spring_layout(g), with_labels=True, font_size=14,
            node_color=list(node_colors.values()), font_color='white')
    plt.savefig(location, bbox_inches='tight')

########################
#以下是模型中定义的一些函数
def initial_record():
    if os.path.exists('./data/working/results.txt'):
        os.remove('./data/working/results.txt')
    with open('./data/working/results.txt','w') as f:    
        f.write("Epoch,Time(s),Loss,F1\n")
        
def normalize(feature_matrix):
    mean = torch.mean(feature_matrix, axis=0)
    stdev = torch.sqrt(torch.sum((feature_matrix - mean)**2, axis=0)/feature_matrix.shape[0])
    return mean, stdev, (feature_matrix - mean) / stdev

def get_f1_score(y_true, y_pred):
    cf_m = confusion_matrix(y_true, y_pred)
    precision = cf_m[1,1] / (cf_m[1,1] + cf_m[0,1] + 10e-5)
    recall = cf_m[1,1] / (cf_m[1,1] + cf_m[1,0])
    f1 = 2 * (precision * recall) / (precision + recall + 10e-5)
    return precision, recall, f1

def evaluate(model, g, features, labels, device):
    "Compute the F1 value in a binary classification case"
    preds = model(g, features.to(device))
    preds = torch.argmax(preds, axis=1).numpy()
    precision, recall, f1 = get_f1_score(labels, preds)
    return f1

def get_model_class_predictions(model, g, features, labels, device, threshold=None):
    unnormalized_preds = model(g, features.to(device))
    pred_proba = torch.softmax(unnormalized_preds, dim=-1)
    if not threshold:
        return unnormalized_preds.argmax(axis=1).detach().numpy(), pred_proba[:,1].detach().numpy()
    return numpy.where(pred_proba.detach().numpy() > threshold, 1, 0), pred_proba[:,1].detach().numpy()

def save_model(g, model, model_dir, id_to_node, mean, stdev):
    # Save Pytorch model's parameters to model.pth
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
    # Save graph's structure information to metadata.pkl for inference codes to initialize RGCN model.
    etype_list = g.canonical_etypes
    ntype_cnt = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}
    with open(os.path.join(model_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump({'etypes': etype_list,
                    'ntype_cnt': ntype_cnt,
                    'feat_mean': mean,
                    'feat_std': stdev}, f)
    # Save original IDs to Node_ids, and trained embedding for non-target node type
    # Covert id_to_node into pandas dataframes
    for ntype, mapping in id_to_node.items():
        # ignore target node
        if ntype == 'target':
            continue
        # retrieve old and node id list
        old_id_list, node_id_list = [], []
        for old_id, node_id in mapping.items():
            old_id_list.append(old_id)
            node_id_list.append(node_id)
        # retrieve embeddings of a node type
        node_feats = model.embed[ntype].detach().numpy()
        # get the number of nodes and the dimension of features
        num_nodes = node_feats.shape[0]
        num_feats = node_feats.shape[1]
        # create id dataframe
        node_ids_df = pandas.DataFrame({'~label': [ntype] * num_nodes})
        node_ids_df['~id_tmp'] = old_id_list
        node_ids_df['~id'] = node_ids_df['~label'] + '-' + node_ids_df['~id_tmp']
        node_ids_df['node_id'] = node_id_list
        # create feature dataframe columns
        cols = {'val' + str(i + 1) + ':Double': node_feats[:, i] for i in range(num_feats)}
        node_feats_df = pandas.DataFrame(cols)
        # merge id with feature, where feature_df use index
        node_id_feats_df = node_ids_df.merge(node_feats_df, left_on='node_id', right_on=node_feats_df.index)
        # drop the id_tmp and node_id columns to follow the Grelim format requirements
        node_id_feats_df = node_id_feats_df.drop(['~id_tmp', 'node_id'], axis=1)
        # dump the embeddings to files
        node_id_feats_df.to_csv(os.path.join(model_dir, ntype + '.csv'),
                                index=False, header=True, encoding='utf-8')

def get_model(ntype_dict, etypes, hyperparams, in_feats, n_classes, device):
    model = HeteroRGCN(ntype_dict, etypes, in_feats, hyperparams['n_hidden'], n_classes, hyperparams['n_layers'], in_feats)
    model = model.to(device)
    return model




