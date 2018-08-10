import torch

def sort_sequence(x):
    '''
    input:[batchsize,seq,25088]
    output:并且按照seq的大小给batch排队并返回排好队的tensor以及每个batch长度的序列
    '''
    zero=torch.zeros(25088)
    batchsize=x.size(0)
    maxseqsize=x.size(1)
    length_list=[]
    for d1 in range(batchsize):
        tem=0
        for d2 in range(maxseqsize):
            if not(torch.equal(x[d1][d2],zero)):
                tem=tem+1
        length_list.append(tem)
    new_seq=[]
    new_lengthlist=[]
    for i in range(len(length_list)):
        max_index=length_list.index(max(length_list))
        new_seq.append(x[max_index])
        new_lengthlist.append(length_list[max_index])
        length_list[max_index]=-100
    new_seq=torch.stack(new_seq,dim=0)
    return new_seq,new_lengthlist