import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torchvision
import argparse

#读取数据
images = []
texts = []
tags = []
train_dataframe = pd.read_csv("./train.txt")
pre_trained_model = './bert-base-uncased'
token = BertTokenizer.from_pretrained(pre_trained_model, mirror='bfsu')

#将tag映射成数字
tag_list = {"neutral": 0, "negative": 1, "positive": 2}

for i in range(train_dataframe.shape[0]):
        guid = train_dataframe.iloc[i]['guid']
        tag = train_dataframe.iloc[i]['tag']
        image = Image.open('./data/' + str(guid) + '.jpg')
        image = image.resize((224, 224), Image.LANCZOS)
        image = np.asarray(image, dtype='float32')
        with open('./data/' + str(guid) + '.txt', encoding='gb18030') as f:
            text = f.read()
        images.append(image.transpose(2, 0, 1))
        texts.append(text)
        tags.append(tag_list[tag])

#去掉无用字符
for i in range(len(texts)):
        text = texts[i]
        word_list = text.replace("#", "").split(" ")
        words_result = []
        for word in word_list:
            if len(word) < 1:
                continue
            elif (len(word)>=4 and 'http' in word) or word[0]=='@':
                continue
            else:
                words_result.append(word)
        texts[i] = " ".join(words_result)

class MultimodalDataset():
    def __init__(self, images, texts, tags, token):
        self.images = images
        self.texts = texts
        self.tags = tags
        self.input_ids, self.attention_masks = text_process(self.texts, token)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        image = self.images[idx]
        text = self.texts[idx]
        tag = self.tags[idx]
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        return image, text, tag, input_id, attention_mask
    
image_text_pairs = [(images[i], texts[i]) for i in range(len(texts))]
# 划分训练集和验证集
X_train, X_valid, tag_train, tag_valid = train_test_split(image_text_pairs, tags, test_size=0.2, random_state=1458, shuffle=True)
image_train, text_train = [X_train[i][0] for i in range(len(X_train))], [X_train[i][1] for i in range(len(X_train))]
image_valid, text_valid = [X_valid[i][0] for i in range(len(X_valid))], [X_valid[i][1] for i in range(len(X_valid))]


class MultimodalModel(nn.Module):
    def __init__(self,type_):
        super().__init__()
        self.type_ = type_
        self.text_model = BertModel.from_pretrained('./bert-base-uncased')
        self.image_model = torchvision.models.densenet201(pretrained=True)
        self.text_linear = nn.Linear(768, 128)
        self.image_linear = nn.Linear(1000, 128)
        self.image_ = nn.Linear(128, 1)
        self.text_ = nn.Linear(128, 1)
        self.fc = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, image):
        image_out = self.image_model(image)
        image_out = self.image_linear(image_out)
        image_out = self.relu(image_out)
        image_w = self.image_(image_out)
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_out = text_out.last_hidden_state[:,0,:]
        text_out.view(text_out.shape[0], -1)
        text_out = self.text_linear(text_out)
        text_out = self.relu(text_out)
        text_w = self.text_(text_out)

        #图像加文本
        if(self.type_ == 1):
            last_out = image_w * image_out + text_w * text_out
            last_out = self.fc(last_out)
        #仅图像
        elif(self.type_ == 2):
            last_out = self.fc(text_out)
        #仅文本
        else:
            last_out = self.fc(image_out)
        return last_out

# 文本预处理
def text_process(text, token):
    result = token.batch_encode_plus(batch_text_or_text_pairs=text, truncation=True, padding='max_length', max_length=32,
                                     return_tensors='pt')
    input_ids = result['input_ids']
    attention_mask = result['attention_mask']
    return input_ids, attention_mask

def train_process(model, train_dataloader, valid_dataloader, train_c, valid_c):
    Loss_C = nn.CrossEntropyLoss()
    #accuracy
    train_a = []
    valid_a = []
    for epoch in range(epoch_num):
        loss = 0.0
        #temp
        train_t = 0
        valid_t = 0
        for idx__, (image, text, target, idx, mask) in enumerate(train_dataloader):
            image, mask, idx, target = image.to(device), mask.to(device), idx.to(device), target.to(device)
            output = model(idx, mask, image)
            optimizer.zero_grad()
            loss = Loss_C(output, target)
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1)
            train_t += int(pred.eq(target).sum())
        train_a.append(train_t / train_c)
        for image, text, target, idx, mask in valid_dataloader:
            image, mask, idx, target = image.to(device), mask.to(device), idx.to(device), target.to(device)
            output = model(idx, mask, image)
            pred = output.argmax(dim=1)
            valid_t += int(pred.eq(target).sum())
        valid_a.append(valid_t / valid_c)
        print('Train Epoch: {}, Train_Loss: {:.4f}, Train Accuracy: {:.4f}, Valid Accuracy: {:.4f}'.format(epoch + 1, loss.item(), train_t / train_c, valid_t / valid_c))

#命令行参数
parser = argparse.ArgumentParser()
#option选1代表既输入文本也输入图像，2代表仅输入图像，3代表仅输入文本
parser.add_argument('--option', type=int, default=1) 
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

type_ = args.option
model = MultimodalModel(type_)
model = model.to(device)
epoch_num = 5
lr = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#数据集
train_dataset = MultimodalDataset(image_train, text_train, tag_train, token)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataset = MultimodalDataset(image_valid, text_valid, tag_valid, token)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)

#训练模型
train_process(model, train_dataloader, valid_dataloader, len(X_train), len(X_valid))

# 预测
result_list = ["neutral", "negative", "positive"]
test = pd.read_csv("./test_without_label.txt")
guid_list = test['guid'].tolist()
tag_pre_list = []
for num in guid_list:
        image = Image.open('./data/' + str(num) + '.jpg')
        image = image.resize((224,224), Image.LANCZOS)
        image = np.asarray(image, dtype = 'float32')
        image = image.transpose(2,0,1)
        with open('./data/' + str(num) + '.txt', encoding='gb18030') as file:
            text = file.read()
        input_id, mask = text_process([text],token)
        image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
        pred = model(input_id.to(device), mask.to(device), torch.Tensor(image).to(device))
        tag_pre_list.append(result_list[pred[0].argmax(dim=-1).item()])
    
result_df = pd.DataFrame({'guid':guid_list, 'tag':tag_pre_list})
result_df.to_csv('./result.txt',sep=',',index=False)