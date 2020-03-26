import torch
from torch import nn, optim

class Server():
    def __init__(self, ID):      
        self.ID = ID
        device_name = "cuda:"+str(ID)
        device = torch.device(device_name)
        output_size = 10
        self.base_model = nn.Sequential(
                    nn.Linear(120, 84),
                    nn.ReLU(True),
                    nn.Linear(84, output_size),
                    nn.LogSoftmax(dim=1)
                )
        self.model = nn.Sequential(
                    nn.Linear(120, 84),
                    nn.ReLU(True),
                    nn.Linear(84, output_size),
                    nn.LogSoftmax(dim=1)
                )
        self.model = self.model.to(device)
        self.opt = optim.SGD(self.model.parameters(), lr=0.03)
    
    #get model on GPU
    def getModel(self):
        return self.model
    
    #get model copy and put it on CPU    
    def getModelCopy(self):
        model_copy = self.base_model
        model_copy.load_state_dict(self.model.state_dict())
        return model_copy
        
    def zero_grads(self):
        self.opt.zero_grad()
    
    def forward(self, prev_out):              
        self.cut_in = prev_out.detach().requires_grad_() 
        pred = self.model(self.cut_in)
        return pred
    
    def backward(self):
        grad_in = self.cut_in.grad
        return grad_in
        
    def step(self):
        self.opt.step()
        
    def eval(self):
        self.model.eval()
        
    def adjust_learning_rate(self, epoch):
        lr = 0.1 * (0.5 ** (epoch // 10))
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr