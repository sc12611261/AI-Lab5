# 多模态情感分析实验
本仓库存储了多模态情感分析实验相关的数据文件和代码文件
## 环境配置
需要安装以下库:
```
pandas==1.4.4
numpy==1.23.3
scikit-learn
torch==2.0.0
torchvision==0.15.1
transformers==4.30.2
```
在终端进入代码所在的文件夹并运行
```
pip install -r requirements.txt
```
## 文件结构
```
|-- data # 实验数据集
|-- main.py # 实验代码
|-- requirement.txt # 执行代码所需要的环境
|-- train.txt # 训练数据文件
|-- test_without_label.txt # 需要预测的文件
|-- result.txt # 结果文件
```
## 执行代码的流程
在终端进入代码所在文件夹并运行
```
python main.py --option=1
```
option选1代表输入文本也输入图像，2代表仅输入图像，3代表仅输入文本
## 参考
1. BERT模型
2. DENSENET201模型