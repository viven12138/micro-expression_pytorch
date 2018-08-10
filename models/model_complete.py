import LSTM
import vggnet
import torch.nn
import torch
import basic_model

#input[batchsize,seq,3,224,224]
#output[batchsize,5]
'''
接受固定大小的seq长度的tensor的输入
返回固定大小的batchsize长度的tensor的输出以及其分数向量
'''
class model_complete(basic_model.basic_model):
    def __init__(self):
        super().__init__()
        self.vgg=vggnet.vggnet_fixed()
        self.LSTM=LSTM.Lstm_fixed()
    def forward(self,x):
        #[batchsize,seq,3,224,224]
        x=self.vgg(x)
        #[seq,batchsize,25088]
        x=self.LSTM(x)
        #[batchsize,5]
        return x
