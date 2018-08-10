import config as config
import utils.get_framesNum as get_framesNum
import utils.video2frames as video2frames
from models import model_complete as model_complete
from data import dataset as dataset
import torch.nn as nn
import torch
import numpy
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from tqdm import tqdm
from utils import visualizer as visualizer

#训练集中加个load
#考虑batch不全的事情（健壮性）

#测试集
def test():
    '''
    计算模型在测试集上的准确率
    '''
    #model
    model=model_complete.model_complete().eval()
    model.load(config.checkpoints_path)
    if torch.cuda.is_available():
        model.cuda()
    #data
    test_data=dataset.Dataset_my(config.test_root,test=True)
    test_dataloader=DataLoader(test_data,batch_size=config.batch_size,shuffle=False)
    #start test
    num_correct=0
    for i1,(data,label) in tqdm(enumerate(test_dataloader)):
        input=Variable(data,volidate=True)
        if torch.cuda.is_available():
            input=input.cuda()
        score=model_complete(input)
        _,pre=torch.max(score,1)
        num_correct=num_correct+(pre==label).sum()
    return float(num_correct)/float(test_data.__len__())

#训练集
def train():
    '''
    训练集合上训练模型
    '''

    if torch.cuda.is_available():
        model=model_complete.model_complete().cuda()
    else:
        model=model_complete.model_complete()
    #data
    train_data=dataset.Dataset_my(config.train_root,train=True)
    val_data=dataset.Dataset_my(config.train_root,train=False)
    train_dataloader=DataLoader(train_data,config.batch_size,shuffle=True)
    val_dataloader=DataLoader(val_data,config.batch_size,shuffle=False)
    #criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr=config.lr
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=config.weight_decay)
    #train
    previous_loss=1e100
    i=0
    for epoch in range(config.epoch):
        times=0
        for ii,(data,label) in tqdm(enumerate(train_dataloader)):
            input=Variable(data)
            target=Variable(label)
            if torch.cuda.is_available():
                input=input.cuda()
                target=target.cuda()
            optimizer.zero_grad()
            score=model(input)
            _, pre = torch.max(score, 1)
            loss=criterion(score,target)
            loss.backward()
            optimizer.step()

            #vis
            print('epoch:{}/ii:{}/'.format(epoch,ii),'loss:',loss)
            # print('vis')
            # print(loss)
            # Visualizer.add_loss(torch.Tensor([i]),loss)
            # i=i+1
            # Visualizer.add_text1(epoch,times,loss)
            # times=times+1
            # print('vis_end')
        model.save()

        #validate and visualizer
        val_loss,val_accuracy=val(model,val_data,val_dataloader)
        print('epoch:'+str(epoch)+'/','val_accuracy:'+str(val_accuracy)+'/','val_loss:'+str(val_accuracy))
        #update learning-rate
        val_loss=float(val_loss)
        if val_loss>previous_loss:
            lr=lr*config.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss=val_loss


def val(model,val_data,val_dataloader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    all_num=val_data.__len__()
    num_correct=0
    loss=0
    for ii,data in tqdm(val_dataloader):
        input,label=data
        input=Variable(input,volatitle=True)
        if torch.cuda.is_available():
            input=input.cuda()
        score = model(input)
        _, pre = torch.max(score, 1)
        num_correct = num_correct + (pre == label).sum()
        loss=criterion(pre,label)+loss
    average_loss=(float(loss)/float(ii+1))
    average_accuracy=(float(num_correct)/float(all_num))
    return average_loss,average_accuracy


train()








