import torch.nn as nn
import time
import config
import torch
checkpoints_path=config.checkpoints_path

class basic_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name=str(type(self))
    def load(self,path):
        '''加载功能
        '''
        self.load_state_dict(torch.load(path))
    def save(self,name=None):
        '''保存功能
        '''
        if name is None:
            prefix = checkpoints_path + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name






