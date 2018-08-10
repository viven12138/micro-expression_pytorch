import utils.video2frames as video2frames
import os
from PIL import  Image
from torch.utils import data
from torchvision import transforms as T
import shutil
import config
import utils.get_framesNum as get_framesNum
import numpy as np
import torch

WORKPLACE=config.workplace
label_rule=config.label_rule

#圖片大小[3,224,224]

class Dataset_my(data.Dataset):
    def __init__(self,root,transforms=None,train=True,test=False):
        '''
               主要目标： 获取所有视频的地址，并根据训练，验证，测试划分数据
        '''
        self.max_num_dic=get_framesNum.find_data_max(root)
        biggest=0
        for key,value in self.max_num_dic.items():
            if biggest<value:
                biggest=value
        self.max_num=biggest
        self.test=test
        self.avs=[]
        for path_son in os.listdir(root):
            for path_gradson in os.listdir(root + '\\' + path_son):
                complete_root = root + '\\' + path_son + '\\' + path_gradson
                self.avs.append(complete_root)
        imgs_num=len(self.avs)
        if self.test:
            self.images=self.avs#返回所有测试集地址的列表
        elif train:
            self.images = self.avs[:int(0.7 * imgs_num)]#返回所有训练集地址的列表
        else:
            self.images = self.avs[int(0.7 * imgs_num):]#返回所有验证集地址的列表
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        '''
            一次返回一个视频的数据，以data_jpgs,label的形式，data_jpgs是一个视频中所有帧的tensor
            ，label是这个视频的标签，具体代表什么表情以label_rule中的标准为准
        '''
        avs_path = self.images[index]
        u = avs_path.split('\\')
        key = u[-2]
        label = label_rule[key]
        data_jpges_list = []
        # 将avs_path的视频转化为照片存储在workplace相应的文件夹下
        video2frames.make_frames(avs_path)
        # 将照片以tensor的列表形式存储在内存中
        for jpgs in os.listdir(WORKPLACE):
            data = Image.open(WORKPLACE + '\\' + jpgs)
            data = self.transforms(data)
            data=data.view(3,224,224)
            data_jpges_list.append(data)
        c=len(data_jpges_list)
        tem=data_jpges_list[0]
        compensate=torch.zeros_like(tem)
        for com in range(self.max_num-c):
            data_jpges_list.append(compensate)
        data_jpgs=torch.stack(data_jpges_list,0)
        shutil.rmtree(WORKPLACE)
        os.makedirs(WORKPLACE)
        return data_jpgs, label
    def __len__(self):
        return len(self.images)




