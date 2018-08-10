import torch
import numpy

def get_seqLast(x,list):
    '''
    input[seq,batchsize,4096]
    output[batchsize,4096](tensor)
    '''
    result=torch.zeros(x.size(1),4096)
    x=x.permute(1,0,2)
    for i,data in enumerate(list):
        result[i]=x[i][data-1]
    return result

