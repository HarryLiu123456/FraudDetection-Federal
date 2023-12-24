# 环境配置指令
1. 以下指令Linux Ubuntu服务器能行，其他不知
```
conda install python==3.7
conda install -c fastchan pytorch==2.0.1

conda install -c dglteam dgl==1.1.3
conda install -c esri matplotlib==3.4.3
conda install -c intel pandas==1.5.3
conda install -c cctbx202211 scikit-learn==1.2.0
```
2. 下pytorch时会下载networkx、numpy

# 表头说明
1. Transaction Table
```
TransactionDT: timedelta from a given reference datetime (not an actual timestamp)
TransactionAMT: transaction payment amount in USD
ProductCD: product code, the product for each transaction
card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.
addr: address
dist: distance
P_ and (R__) emaildomain: purchaser and recipient email domain
C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
D1-D15: timedelta, such as days between previous transaction, etc.
M1-M9: match, such as names on card and address, etc.
Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.
```
2. Categorical Features:
```
ProductCD
card1 - card6
addr1, addr2
P_emaildomain
R_emaildomain
M1 - M9
```
3. Identity Table
```
Variables in this table are identity information – network connection information (IP, ISP, Proxy, etc) and digital signature (UA/browser/os/version, etc) associated with transactions.
They're collected by Vesta’s fraud protection system and digital security partners.
(The field names are masked and pairwise dictionary will not be provided for privacy protection and contract agreement)
Categorical Features:
DeviceType
DeviceInfo
id_12 - id_38
```

# 程序设计说明

1. 节点的构建：
- 每个交易或用户身份由train_identity.csv中的TransactionID表示，作为图中的唯一节点。
- 节点属性来自数据集的其他列，如id_01到id_38，DeviceType和DeviceInfo。这些属性包含了与用户身份和设备相关的信息。

2. 节点属性的分配：
- 在GNN模型中，节点属性可以通过数据文件直接传入或者使用可训练的嵌入（对于没有直接特征的节点类型）。
- 例如，在HeteroRGCN类的实现中，每个节点类型的嵌入通过nn.Parameter初始化，然后在模型的forward方法中用作节点特征。

3. 边的构建：
- 边的构建基于交易之间的关系。再train_trasaction.csv文件里读取的
- 如果两个交易共享同一个信用卡号（如card1），则可以在对应的节点之间创建一条边。
- 在代码中，边的构建通过解析边列表文件来完成。每一行表示一个边，通过parse_edgelist函数读取。

4. 如何确定节点是否相连：
- 节点间的连接通过交易数据中的共享属性来确定。例如，如果两个用户的交易记录在train_transaction.csv中具有相同的地址或邮箱域，则这两个用户的节点在图中通过边相连。
- 这种关系在construct_graph函数中实现，其中根据edgelists参数中的边列表来构建图。

5. 在client中定义了一个差异字典，键对应状态字典中的键，值是当前用户局部网络的值与全局网络的值的差

6. 在server中定义了一个聚合操作，将上面诸个client的差异字典的值取平均，然后加到全局网络的值上

7. 因为直接将全局网络状态字典的键和值做了一个对应，所以不需要考虑全局网络参数状态字典的样子，但是超网络需要生成网络的参数，所以需要知道网络参数都有哪些键和值，但是这个我还没弄清

8. 做欺诈检测的是一个异构图（只有结点作为训练对象），然后用图卷积神经网络（结点与相邻结点之间有个全连接层，卷积就是相同类型的边复用一个全连接层）

# 原版
```json
{
    "training_dir" : "./data/working", 
    "model_dir" : "./data/working",
    "output_dir" : "./data/working",
    "nodes" : "features.csv",
    "target_ntype" : "TransactionID",
    "edges" : "./data/working/relation*",
    "labels" : "tags.csv",
    "new_accounts" : "test.csv",
    "compute_metrics" : true,
    "threshold" : 0,
    "num_gpus" : 0,
    "optimizer" : "adam",
    "lr" : 1e-2,
    "n_hidden" : 32,
    "n_layers" : 6,
    "weight_decay" : 5e-4,
    "dropout" : 0.2,
    "embedding_size" : 360,

    "number_clients" : 10,
    "local_epochs" : 3,
    "global_epochs" : 5,
    "k" : 2,
    "lambda" : 0.1,

    "flag" : false
}
```

# 其他说明（作废）
1. 在FraudDetection目录下运行以下代码以自动获取所需模块，建议在虚拟环境中运行
    ```
    pip install -r requirements.txt
    ```
    如果是conda建议逐个安装
2. 需要单独安装cuda版本dgl模块，输入如下指令
    ```
    pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
    pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
    ```
    或者
    ```
    conda install -c dglteam/label/cu118 dgl
    ```
3. 在FraudDetection目录下运行以下代码以运行模型
    ```
    python ./main/main.py
    ```
    或者
    ```
    python ./main/main.py --conf conf.json
    ```
4. conf.json中存储了模型的一些参数可供调整

# 补丁（作废）
1. 在conf.json将"labels"、"new_accounts"改成了相对路径
2. 在conf.json中加入"flag1"、"flag2"用于快速调试程序
3. 在utils.py的50行加入了处理错误的函数
4. 增加了将图放在了device上


