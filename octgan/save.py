import os
import torch

def make_dir():
    path = './result/'
    os.makedirs(path)
    for i in ['/csv','/logs','/model']:
        os.makedirs(path+i)

def save_model(generator, epoch):
    if not os.path.exists('./result/'):
        make_dir()

    PATH = "./result/model/"
    
    model_dict = generator.state_dict()

    PATH += "{}.pth".format(epoch)
    torch.save(model_dict, PATH)
    