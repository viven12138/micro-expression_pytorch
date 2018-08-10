import torch
import torch.nn as nn
import  basic_model
import utils.sort_sequence as sort_sequence
from torch.nn import utils as nn_utils
import utils.get_seqLast as get_seqLast

#input[seq,batchsize,25088] 该为0的仍然为0
#output[batchsize,5]
class Lstm_fixed(basic_model.basic_model):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Sequential(
            #[seq,batchsize,25088]
            nn.LSTM(25088,4096,num_layers=1),#[seq,batchsize,4096]
        )
        self.layer2=nn.Sequential(
            #[batchsize,4096]
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,5),
            nn.BatchNorm1d(5),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Softmax(),
            #[batchsize,5]
        )
    def forward(self,x):
        #[seq,batchsize,25088]
        x=x.permute(1,0,2)
        #[batchsize,seq,25088]
        x,seqLength=sort_sequence.sort_sequence(x)
        #pack it
        pack=nn_utils.rnn.pack_padded_sequence(x,seqLength,batch_first=True)
        #feed lstm
        out,_=self.layer1(pack)
        #unpacked
        unpacked,_=nn_utils.rnn.pad_packed_sequence(out)
        #[seq,batchsize,4096]
        result=get_seqLast.get_seqLast(unpacked,seqLength)
        #[batchsize,4096]
        result=self.layer2(result)
        #[batchsize,5]
        return result
        #[batchsize,5]


