# 说明
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

# 补丁
1. 在conf.json将"labels"、"new_accounts"改成了相对路径
2. 在conf.json中加入"flag1"、"flag2"用于快速调试程序
3. 在utils.py的50行加入了处理错误的函数
4. 增加了将图放在了device上