import torch
from torch import nn, optim

class Client():
    def __init__(self, ID):
        self.ID = ID
        device_name = "cuda:"+str(ID)
        device = torch.device(device_name)
        self.base_model = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),            
            nn.MaxPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.ReLU(True),
        )
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),            
            nn.MaxPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.ReLU(True),
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
        
    def getID(self):
        return self.ID
        
    def zero_grads(self):
        self.opt.zero_grad()
        
    def forward(self, x):
        self.output = self.model(x)
        return self.output
        
    def backward(self, grad_in):
        return self.output.backward(grad_in)
        
    def step(self):
        self.opt.step()
    
    def eval(self):
        self.model.eval()
    
    def toGpu(self):
        gpu = torch.device("cuda:"+str(self.ID))
        self.model.to(gpu)