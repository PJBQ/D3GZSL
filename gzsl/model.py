from pydoc import classname
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(opt.attr_dim + opt.nz, 4096)
        self.fc2 = nn.Linear(4096, opt.x_dim)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.apply(weights_init)
    def forward(self, noise, att):
        o = torch.cat((noise, att), 1)
        o = self.lrelu(self.fc1(o))
        o = self.relu(self.fc2(o))

        return o

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(opt.x_dim + opt.attr_dim, 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        o = torch.cat((x, att), 1)
        o = self.lrelu(self.fc1(o))
        o = self.fc2(o)

        return o

class  CLS_MLP(nn.Module):
    def __init__(self, opt):
        super(CLS_MLP, self).__init__()
        self.fc1 = nn.Linear(opt.x_dim, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.n_class_all)
        
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        embedding= self.relu(self.fc1(x))
        o = self.logic(self.fc2(embedding))

        return o

class F_MLP(nn.Module):
    def __init__(self, opt):
        super(F_MLP, self).__init__()
        self.fc1 = nn.Linear(opt.embedSize, opt.n_class_all)
        self.fc2 = nn.Linear(opt.n_class_all, 2)
        self.fc3 = nn.Linear(opt.embedSize, 2)
        
        self.relu = nn.ReLU(True)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softMax = nn.Softmax(dim=1)
    
    def forward(self, x, seenclasses, unseenclasses):
        embedding = self.fc1(x)
        o = self.log_softmax(embedding)
        bi_feature = self.fc2(o)
        # bi_feature = self.fc3(x)
        kl_input = self.get_kl_input(embedding, seenclasses)
        # kl_input = F.normalize(kl_input, p=2.0, dim=1, eps=1e-12, out=None)
        kl_input = self.softMax(kl_input)
        return o, bi_feature, kl_input


    def get_kl_input(self, _embed, seenclasses):
        seenclasses = seenclasses.cuda()
        Y_seen = torch.index_select(_embed, 1, seenclasses)
        return Y_seen


class OOD_MLP(nn.Module):
    def __init__(self, opt):
        super(OOD_MLP, self).__init__()
        self.fc1 = nn.Linear(opt.x_dim, opt.ood_embedSize)
        self.fc2 = nn.Linear(opt.ood_embedSize, opt.n_class_seen)
        # self.fc3 = nn.Linear(opt.ood_embedSize, opt.ood_embedSize)
        
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)
        # self.softmax = nn.Softmax()
    
    def forward(self, x):
        embedding= self.relu(self.fc1(x))
        embedding = self.fc2(embedding)
        # sx = self.softmax(embedding)
        o = self.logic(embedding)
        return o, embedding

class OOD_MLP_DEEP(nn.Module):
    def __init__(self, opt):
        super(OOD_MLP_DEEP, self).__init__()
        # self.fc1 = nn.Linear(opt.x_dim, opt.embedSize)   
        self.fc2 = nn.Linear(opt.embedSize, opt.n_class_seen)
        # self.relu = nn.ReLU(True)
        # self.lrelu = nn.LeakyReLU(0.2, True)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.soft = nn.Softmax(dim=1)
        self.apply(weights_init)
    
    def forward(self, x):
        # embedding = self.relu(self.fc1(x))
        # embedding = self.fc2(embedding)
        embedding = self.fc2(x)
        o = self.log_softmax(embedding)
        # embedding = F.normalize(embedding, p=2.0, dim=1, eps=1e-12, out=None)
        s = self.soft(embedding)
        return o, s


class LearnSigmoid(nn.Module):
    def __init__(self, _weight, _offset):
        super(LearnSigmoid, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.offset = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight.data.fill_(_weight)
        self.offset.data.fill_(_offset)

    # def reset_parameters(self):
    #     self.weight.data.fill_(1.0)
    #     self.offset.data.fill_(0)

    def forward(self, input):
        return 1/(1 + torch.exp(-self.weight*(input + self.offset)))


class B_module(nn.Module):
    def __init__(self, opt):
        super(B_module).__init__()
        self.fc1 = nn.Linear(opt.x_dim, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.n_class_all)
        self.fc3 = nn.Linear(opt.n_class_all, 2)
        
        self.relu = nn.ReLU(True)
        # self.relu2 = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)
        self.softMax = nn.Softmax(dim=1)
    
    def forward(self, _input):
        e = self.relu(self.fc1(_input))
        e = self.relu(self.fc2(e))
        o = self.softMax(self.fc3(e))
        return o

class Embedding_Net(nn.Module):
    def __init__(self, opt):
        super(Embedding_Net, self).__init__()

        self.fc1 = nn.Linear(opt.x_dim, opt.embedSize)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, features):
        embedding= self.relu(self.fc1(features))
        return embedding