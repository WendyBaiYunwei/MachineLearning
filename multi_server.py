import numpy as np
import torch
import torchvision
import time
import copy
import torch.multiprocessing as mp
from m_server import Server
from m_client import Client
from dataset import MyData
from test_dataset import MyTestData
from torchvision import datasets, transforms
from torch import nn, optim

def train(images, labels, client, server, clientID):
    device_name = "cuda:"+str(clientID)
    device = torch.device(device_name)
    
    images, labels = images.to(device), labels.to(device)
    
    client.zero_grads()
    server.zero_grads()
    
    cut_out = client.forward(images)
    pred = server.forward(cut_out)
    
    criterion = nn.NLLLoss()
    loss = criterion(pred, labels)
    
    loss.backward()

    grad_in = server.backward()
    client.backward(grad_in)

    client.step()
    server.step()

        
def test(dataloader, client_i, clients, servers, dataset_name):
    clients[client_i].eval()
    servers[client_i].eval()
    correct = 0
    device_name = "cuda:"+str(client_i)
    device = torch.device(device_name)
    with torch.no_grad():
        for data, target in dataloader:
            data = data.type(torch.FloatTensor)
            target = target.view(target.shape[0]).type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            cut_out = clients[client_i].forward(data)
            pred = servers[client_i].forward(cut_out)
            pred = pred.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            
    return(100. * int(correct) / len(dataloader.dataset))
        
if __name__ == '__main__':
    epochs = 20
    torch.manual_seed(0)
    ttlClients = 4
    #prepare training data
    trainloaders = []
    for i in range(5):
        trainset = MyData('./', i+1)
        trainloaders.append(torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True))
    testloaders = []
    for i in range(ttlClients):
        trainset = MyTestData('./', i+1)
        testloaders.append(torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True))    
        
    #initialize clients and servers
    servers = []
    clients = []
    ttlClients = 4
    for i in range(ttlClients):
        client = Client(i)
        clients.append(client)
        server = Server(i)
        servers.append(server)
                                                   
    #init multiprocessing    
    mp.set_start_method('spawn')
    for epoch in range(epochs):    
        print(epoch)
        processes = []
        #here, I ignore even data split and data completeness. 
        #Just want to achieve multi-processing first
        #Each process is assigned to 1 GPU, and the start-join is carried out
        #after each step
        for client_i, (images, labels) in enumerate(trainloaders[0]):
            print(client_i)
            client_i %= ttlClients
            if client_i == 0 and processes:
                for p_i in range(len(processes)):
                    processes[p_i].start()
                for p_i in range(len(processes)-1,-1,-1):
                    processes[p_i].join()
                processes.clear()
            images = images.type(torch.FloatTensor)
            labels = labels.view(labels.shape[0]).type(torch.LongTensor)
            p = mp.Process(target=train, args=[images, labels, clients[client_i], servers[client_i], client_i])
            processes.append(p)
        
        #deal with leftovers
        if processes:
            for p_i in range(len(processes)):
                processes[p_i].start()
            for p_i in range(len(processes)-1,-1,-1):
                processes[p_i].join()
            processes.clear() 
        
        #express bitter happiness for ending one epoch
        print('hahaha')
        #test
        if epoch%2 == 0:
            curr_acc = test(testloaders[0], 0, clients, servers, "Test set")
            print(curr_acc)
            
        #average the weights after every epoch
        client_weights = []
        server_weights = []
        local_device = torch.device("cpu")
        for client_i in range(ttlClients):
            client_weight = clients[client_i].getModelCopy().to(local_device).state_dict()
            server_weight = servers[client_i].getModelCopy().to(local_device).state_dict()
            client_weights.append(client_weight)
            server_weights.append(server_weight)
        
        def average_weights(w):
            w_avg = w[0]
            for key in w_avg.keys():
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key]
                w_avg[key] = torch.div(w_avg[key], len(w))
            return w_avg
        
        global_client_weights = average_weights(client_weights)
        global_server_weights = average_weights(server_weights)
        for client_i in range(ttlClients):
            clients[client_i].getModel().load_state_dict(global_client_weights)
            servers[client_i].getModel().load_state_dict(global_server_weights)
    
    #final test
    overall_acc = 0
    for i in range(ttlClients):   
        curr_acc = test(testloaders[i], i, clients, servers, "Test set")
        print(curr_acc)
        overall_acc += curr_acc

    print(overall_acc/ttlClients)