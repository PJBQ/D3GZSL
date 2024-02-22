import torch.nn as nn
import torch
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def reparameter(mu,sigma):
    return (torch.randn_like(mu) *sigma) + mu

class Embedding_Net(nn.Module):
    def __init__(self, opt):
        super(Embedding_Net, self).__init__()

        self.fc1 = nn.Linear(opt.resSize, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.outzSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, features):
        embedding= self.relu(self.fc1(features))
        out_z = F.normalize(self.fc2(embedding), dim=1)
        return embedding,out_z

class Embedding_OOD_Net(nn.Module):
    def __init__(self, opt):
        super(Embedding_OOD_Net, self).__init__()

        self.fc1 = nn.Linear(opt.x_dim, opt.embedSize)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, features):
        embedding= self.relu(self.fc1(features))
        return embedding

class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class MLP_CRITIC(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h


class Dis_Embed_Att(nn.Module):
    def __init__(self, opt):
        super(Dis_Embed_Att, self).__init__()
        self.fc1 = nn.Linear(opt.embedSize+opt.attSize, opt.nhF)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.nhF, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, input):
        h = self.lrelu(self.fc1(input))
        h = self.fc2(h)
        return h

# my Dis_Embed_Att
class My_Dis_Embed_Att(nn.Module):
    def __init__(self, opt):
        super(Dis_Embed_Att, self).__init__()
        self.fc1 = nn.Linear(opt.embedSize+opt.attSize, opt.nhF)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.nhF, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, input):
        h = self.lrelu(self.fc1(input))
        h = self.fc2(h)
        #
        return h

class OOD_MLP(nn.Module):
    def __init__(self, opt):
        super(OOD_MLP, self).__init__()
        self.fc1 = nn.Linear(opt.x_dim, opt.ood_embedSize)
        self.fc2 = nn.Linear(opt.ood_embedSize, opt.n_class_seen)
        
        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        embedding= self.relu(self.fc1(x))
        embedding = self.fc2(embedding)
        o = self.logic(embedding)
        return o, embedding

class OOD_MLP_DEEP(nn.Module):
    def __init__(self, opt):
        super(OOD_MLP_DEEP, self).__init__()
        self.fc2 = nn.Linear(opt.embedSize, opt.n_class_seen)
        # self.relu = nn.ReLU(True)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.soft = nn.Softmax(dim=1)
        self.apply(weights_init)
    
    def forward(self, x):

        embedding = self.fc2(x)
        o = self.log_softmax(embedding)
        # embedding = F.normalize(embedding, p=2.0, dim=1, eps=1e-12, out=None)
        s = self.soft(embedding)
        return o, s

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