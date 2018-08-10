from torchvision import models
import torch.nn as nn
import basic_model
import torch

#input:[batchsize,seq,3,224,224]
#output[seq,batchsize,25088]
#稳了
class vggnet_fixed(basic_model.basic_model):
    def __init__(self,model=models.vgg16(pretrained=True)):
        super().__init__()
        self.vgg=nn.Sequential(*list(model.children())[:-1])
        for p in self.parameters():
            p.requires_grad=False
    def forward(self,x):
        x=x.permute(1,0,2,3,4)
        #[seq,batchsize,3,224,224]
        size_seq=x.size(0)
        size_batchsize=x.size(1)
        zero_1=torch.zeros_like(x[0][0])
        #全0 [3,224,224]tensor
        result_x=[]
        point=True
        for d1 in range(size_seq):
            tem=0
            new_batch_list=[]
            for d2 in range(size_batchsize):
                if not(torch.equal(x[d1][d2],zero_1)):
                    new_batch_list.append(x[d1][d2])
                    tem=tem+1
            if tem==0:
                result_x.append(torch.stack([zero_2]*size_batchsize,0))
                continue
            new_batch_result=torch.stack(new_batch_list,dim=0)
            #[batchsize*,3,224,224]
            new_batch_result=self.vgg(new_batch_result)
            #[batchsize*,512,7,7]
            if point:
                zero_2=torch.zeros_like(new_batch_result[0])#全0的[512,7,7]
                point=False
            space_num=size_batchsize-tem
            if space_num==0:
                result_x.append(new_batch_result)
            else:
                space = [zero_2] * (space_num)
                space = torch.stack(space, 0)
                result_x.append(torch.cat((new_batch_result, space),dim=0))
        result_tensor=torch.stack(result_x,0)
        #[seq,batchsize,512,7,7]
        result_tensor=result_tensor.view(size_seq,size_batchsize,-1)
        return result_tensor
        #[seq,batchsize,25088]
