import os
import time
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import laplace, norm
from sklearn import preprocessing
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset


class DATA_LOADER(Dataset):
    def __init__(self, opt, is_train=True, is_seen=False, is_syn=False, syn_feature=None, syn_label=None):
        self.is_train = is_train
        self.is_seen = is_seen
        self.is_syn = is_syn
        self.syn_feature = syn_feature
        self.syn_label = syn_label
        self.read_matdataset(opt)    
        if self.is_syn:
            self.train_feature = torch.cat((self.train_feature, self.syn_feature), 0)
            self.train_label = torch.cat((self.train_label, self.syn_label), 0)
            self.ntrain = self.train_feature.size()[0]
        self.map_label = map_label_all(self.train_label, self.seenclasses, self.unseenclasses, self.ntrain_class)
        self.map_test_seen_label = map_label_all(self.test_seen_label, self.seenclasses, self.unseenclasses, self.ntrain_class)
        self.map_test_unseen_label = map_label_all(self.test_unseen_label, self.seenclasses, self.unseenclasses, self.ntrain_class)


    def map_label_all(self, label, seenclasses, unseenclasses, _nclass_s):
        mapped_label = torch.LongTensor(label.size())
        nclass_s = _nclass_s
        for i in range(seenclasses.size(0)):
            mapped_label[label == seenclasses[i]] = i
        
        for j in range(unseenclasses.size(0)):
            mapped_label[label == unseenclasses[j]] = j + nclass_s
            
        return mapped_label

    def read_matdataset(self, opt):
        # resnet101
        matcontent = sio.loadmat(opt.res_path)
        feature = matcontent['features'].T
        self.all_file = matcontent['image_files']
        label = matcontent['labels'].astype(int).squeeze() - 1

        # index
        matcontent = sio.loadmat(opt.att_path)
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        
        # attribute
        self.attribute = torch.from_numpy(matcontent['att'].T).float()

        if not opt.validation:
            self.train_image_file = self.all_file[train_loc]
            self.test_seen_image_file = self.all_file[test_seen_loc]
            self.test_unseen_image_file = self.all_file[test_unseen_loc]

            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
            
                # feature
                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])

                # train
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long()

                # test unseen
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()

                # test seen
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()
            
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.attribute_seen = self.attribute[self.seenclasses]

        self.train_samples_class_index = torch.tensor([self.train_label.eq(i_class).sum().float() for i_class in self.train_class])

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        batch_map_label = self.map_label[idx]
        return batch_feature, batch_label, batch_att, batch_map_label
        
    def __getitem__(self, index):
        if self.is_train:
            feature = self.train_feature[index]
            label = self.train_label[index]
            map_label = self.map_label[index]
            att = self.attribute[label]
            return feature, label, att, map_label
        else:
            if self.is_seen:
                feature = self.test_seen_feature[index]
                label = self.test_seen_label[index]
                att = self.attribute[label]
                return feature, label, att, label   # test no map label
            else:
                feature = self.test_unseen_feature[index]
                label = self.test_unseen_label[index]
                att = self.attribute[label]
                return feature, label, att, label   # test no map label

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            if self.is_seen:
                return len(self.test_seen_label)
            else:
                return len(self.test_unseen_label)

    
# def val_gzsl(test_label, predicted_label, target_classes):
#     acc_per_class = 0
#     for i in target_classes:
#         idx = (test_label == i)
#         acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
#     acc_per_class /= target_classes.size(0)
#     return acc_per_class

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def fig_ood_loss(map_label, logit_v, output, ood_criterion,opt):
    # 分割ood数据与id数据
    id_output = torch.FloatTensor().cuda()
    id_label = torch.LongTensor().cuda()
    # id_logit = torch.FloatTensor().cuda()
    ood_logit = torch.FloatTensor().cuda()

    idj = (map_label < opt.n_class_seen)
    id_label = torch.cat((id_label, map_label[idj]), 0)
    # id_logit = torch.cat((id_logit, logit_v[idj]), 0)
    id_output = torch.cat((id_output, output[idj]), 0)

    # loss
    class_loss = opt.oodtrain_loss_para*ood_criterion(id_output, id_label)

    # ood
    # idx = (map_label >= opt.n_class_seen)
    # ood_logit = torch.cat((ood_logit, logit_v[idx]), 0)

    # 计算energy
    # Ec_out = -torch.logsumexp(ood_logit, dim=1)
    # Ec_in = -torch.logsumexp(id_logit, dim=1)
    # ood_loss = opt.ood_loss_para*(torch.pow(F.relu(Ec_in-opt.m_in), 2).mean() + torch.pow(F.relu(opt.m_out-Ec_out), 2).mean())
    # return class_loss, ood_loss
    return class_loss

def bi_acc(pre_label, map_label, opt):
        real_label = torch.LongTensor(len(map_label)).cuda()
        real_label.copy_(map_label)

        real_label[(real_label < opt.n_class_seen)] = 0
        real_label[(real_label >= opt.n_class_seen)] = 1

        sum_sample = len(real_label)
        correct_sample = torch.sum(pre_label == real_label)
        # acc = float(correct_sample) / float(sum_sample)
        # print('val binary acc: {:.4f}  correct_sample: {:.4f}  sum_sample: {:.4f}'.format(acc, correct_sample, sum_sample))
        return float(correct_sample) / float(sum_sample)

def val_ood(test_X, test_label, val_model, netOODMap, opt, is_ood): 
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size()).cuda()
        for i in range(0, ntest, opt.batch_size):
            end = min(ntest, start+opt.batch_size)
            with torch.no_grad():
                if opt.cuda:
                    embed = netOODMap(test_X[start:end].cuda())
                    output, _ = val_model(embed)
                else:
                    embed = netOODMap(test_X[start:end])
                    output, _ = val_model(embed)
                # energy
                # energy_score = -torch.logsumexp(logit_v, dim=1)
                energy_score = output.max(dim=1).values
                # threshold = opt.threshold
                di_label = torch.LongTensor(energy_score.size()).cuda()
                if is_ood:
                    in_idx = (energy_score > opt.ood_threshold)
                    out_idx = (energy_score <= opt.ood_threshold)
                    di_label[in_idx] = 0
                    di_label[out_idx] = 1
                else:
                    in_idx = (energy_score > opt.id_threshold)
                    out_idx = (energy_score <= opt.id_threshold)
                    di_label[in_idx] = 0
                    di_label[out_idx] = 1
            # print(di_label)
            predicted_label[start:end] = di_label
            # print(predicted_label[start:end])
            # print(predicted_label)
            start = end

        acc = bi_acc(predicted_label, test_label, opt)
        return acc

def val_ood_cls(test_X, test_label, val_model, netOODMap, opt, _start, _end): 
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, opt.test_batch_size):
            end = min(ntest, start+opt.test_batch_size)
            with torch.no_grad():
                if opt.cuda:
                    embed = netOODMap(test_X[start:end].cuda())
                    output, _ = val_model(embed)
                else:
                    embed = netOODMap(test_X[start:end])
                    output, _ = val_model(embed)
            _, predicted_label[start:end] = torch.max(output, 1)
            start = end

        acc = compute_per_class_acc_ood(test_label, predicted_label, _start, _end)
        return acc

def val_gzsl(test_X, test_label, seen_classes, unseen_classes, netMap, val_model, opt, _is_seen = True): 
    if _is_seen:
        target_classes = seen_classes
    else:
        target_classes = unseen_classes
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, opt.batch_size):
            end = min(ntest, start+opt.batch_size)
            with torch.no_grad():
                if opt.cuda:
                    embed = netMap(test_X[start:end].cuda())
                    output, _, _ = val_model(embed, seen_classes, unseen_classes)
                else:
                    embed = netMap(test_X[start:end])
                    output, _, _ = val_model(embed, seen_classes, unseen_classes)
                # output = torch.nn.LogSoftmax(output)
            _, predicted_label[start:end] = torch.max(output, 1)
            start = end

        acc = compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

def val_cls(test_X, test_label, target_classes, val_model, opt): 
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, opt.batch_size):
            end = min(ntest, start+opt.batch_size)
            with torch.no_grad():
                if opt.cuda:
                    output = val_model(test_X[start:end].cuda())
                else:
                    output = val_model(test_X[start:end])
            _, predicted_label[start:end] = torch.max(output, 1)
            start = end

        acc = compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    acc_per_class /= target_classes.size(0)
    return acc_per_class 

def compute_per_class_acc_ood(test_label, predicted_label, start, end):
    acc_per_class = 0
    for i in range(start, end):
        idx = (test_label == i)
        acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    acc_per_class /= (end - start)
    return acc_per_class 

def map_label_all(label, seenclasses, unseenclasses, _nclass_s):
    mapped_label = torch.LongTensor(label.size())
    nclass_s = _nclass_s
    for i in range(seenclasses.size(0)):
        mapped_label[label == seenclasses[i]] = i
    
    for j in range(unseenclasses.size(0)):
        mapped_label[label == unseenclasses[j]] = j + nclass_s
        
    return mapped_label

def generate_syn_feature_for_ood_train(netG, classes, attribute, num, opt):
    # for p in netG.parameters():  # reset requires_grad
    #     print(p.requires_grad)
        # p.requires_grad = True
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.x_dim)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attr_dim)
    syn_noise = torch.FloatTensor(num, opt.nz)

    
    # syn_label = Variable(syn_label, requires_grad=True)
    # syn_att = Variable(syn_att, requires_grad=True)
    # syn_noise = Variable(syn_noise, requires_grad=True)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(syn_noise, syn_att)

        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    # syn_feature.requires_grad = True
    syn_feature = Variable(syn_feature, requires_grad=True)
    # print(syn_feature.grad)
    return syn_feature, syn_label

def generate_test(netG, classes, attribute, num, opt):
    nclass = classes.size(0)
    # syn_feature = torch.FloatTensor()
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(nclass * num, opt.attr_dim)
    syn_noise = torch.FloatTensor(nclass * num, opt.nz)
    # syn_feature = Variable(syn_feature, requires_grad=True)
    
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        # syn_feature = syn_feature.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.narrow(0, i * num, num).copy_(iclass_att.repeat(num, 1))
        syn_label.narrow(0, i * num, num).fill_(iclass)
    syn_noise.normal_(0, 1)
    
    output = netG(syn_noise, syn_att).cpu()
    # syn_feature = torch.cat((syn_feature, output), 1)
    
    return output, syn_label

def print_net_para(_net):
    net = _net
    # 
    # print(net)

    # 
    # print(net.state_dict())

    # 
    # print(net.state_dict().keys())

    # 
    # # print(net.state_dict()["net1.bias"])
    # # print(net.state_dict()["net1.bias"].shape)

    # 
    for name, parameters in net.named_parameters():
        print(name,':', parameters, parameters.size())

class NET_GRAD():
    def __init__(self, _net):
        self.net = _net
        self.grad_dict = {}
        for name, parameters in self.net.named_parameters():
            print(name,':', parameters.grad, parameters.size())
            self.grad_dict[name] = parameters.grad
        


def print_net_grad(_net):
    net = _net
    for name, parameters in net.named_parameters():
        print(name,':', parameters.grad, parameters.size())

# ratio
def ratio(_score_seen, _score_unseen, _score_train, opt, _epoch, _str='default'):
    data_seen = _score_seen.cpu()
    data_seen = data_seen.detach().numpy()
    data_unseen = _score_unseen.cpu()
    data_unseen = data_unseen.detach().numpy()
    data_train = _score_train.cpu()
    data_train = data_train.detach().numpy()
    sns.set_palette("hls")
    plt.figure(dpi=120)
    # sns.set(style='dark')
    # sns.set_style("dark",{"axes.facecolor": "#e9f3ea"})
    # s = sns.distplot(data_seen, 
    #                 hist=True, 
    #                 kde=True, 
    #                 kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c72e29'}, 
    #                 fit=norm, 
    #                 color='#098154', 
    #                 axlabel='Standardized Residual')
    # u = sns.distplot(data_unseen, 
    #                 hist=True, 
    #                 kde=True, 
    #                 kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c12e32'}, 
    #                 fit=norm, 
    #                 color='#542378', 
    #                 axlabel='Standardized Residual')
    s = sns.kdeplot(data=data_seen,color='#ab2821')
    u = sns.kdeplot(data=data_unseen,color='#53a2d8')
    t = sns.kdeplot(data=data_train,color='#232323')
    # now = datetime.now()    # 
    # timestr = now.strftime("%Y_%m_%d_%H_%M_%S")
    file_name = "_" + _str
    path = opt.png_root + str(_epoch) + file_name + opt.png_suffix
    plt.savefig(path)
    # plt.show()

def ratio_f_score(_score_seen, opt, _epoch, _str='default'):
    data_seen = _score_seen.cpu()
    data_seen = data_seen.detach().numpy()
    sns.set_palette("hls")
    plt.figure(dpi=120)
    s = sns.kdeplot(data=data_seen,color='#ab2821')
    # s = sns.lineplot(data=data_seen,color='#ab2821')
    # now = datetime.now()    # 
    # timestr = now.strftime("%Y_%m_%d_%H_%M_%S")
    file_name = "_" + _str
    path = opt.png_root + str(_epoch) + file_name + opt.png_suffix
    plt.savefig(path)

def contruct_double_variable(_seen_score, _unseen_score):
    seen_score = _seen_score.T
    unseen_score = _unseen_score.T

    pass

def get_energy_score(test_X, test_label, val_model, opt):
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        energy_score = torch.FloatTensor(test_label.size()).cuda()
        for i in range(0, ntest, opt.batch_size):
            end = min(ntest, start+opt.batch_size)
            if opt.cuda:
                output, logit_v = val_model(test_X[start:end].cuda())
            else:
                output, logit_v = val_model(test_X[start:end])
            # energy
            # energy_score[start:end] = -torch.logsumexp(logit_v, dim=1)
            # energy_score[start:end] = output.max(dim=1).values
            energy_score[start:end] = logit_v.max(dim=1).values
            # print(energy_score[start:end])
            start = end
        return energy_score

def get_best_acc(acc_seen, acc_unseen, _best_seen, _best_unseen, _best_H):
    best_seen = _best_seen
    best_unseen = _best_unseen
    best_H = _best_H
    if (acc_seen+acc_unseen)==0:
            print('a bug')
            H=0
    else:
        H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
    if H > best_H:
    # if H > best_H and acc_seen > self.opt.paper_seen_acc:
        best_seen = acc_seen
        best_unseen = acc_unseen
        best_H = H
    return best_seen, best_unseen, best_H

def topK(_score_seen, _score_unseen, _socre_train, _largest=True):
    # seen_values, seen_indices = torch.topk(_score_seen, 1764, dim=0, largest=_largest)
    # unseen_values, unseen_indices = torch.topk(_score_unseen, 2967, dim=0, largest=_largest)
    # train_values, train_indices = torch.topk(_socre_train, 7057, dim=0, largest=_largest)
    # seen_values, seen_indices = torch.topk(_score_seen, 1675, dim=0, largest=_largest)
    # unseen_values, unseen_indices = torch.topk(_score_unseen, 2818, dim=0, largest=_largest)
    # train_values, train_indices = torch.topk(_socre_train, 6704, dim=0, largest=_largest)
    seen_values, seen_indices = torch.topk(_score_seen, 1587, dim=0, largest=_largest)
    unseen_values, unseen_indices = torch.topk(_score_unseen, 2670, dim=0, largest=_largest)
    train_values, train_indices = torch.topk(_socre_train, 6351, dim=0, largest=_largest)

    if _largest:
        return seen_values.min().item(), unseen_values.min().item(), train_values.min().item()
    else:
        return seen_values.max().item(), unseen_values.max().item(), train_values.max().item()

def fig_alpha(_score, _opt, _data = 0):
    if _data == 0:
        min_score = _opt.min_score_unseen
        max_score = _opt.max_score_unseen
        mean_score = _opt.mean_score_unseen
        length_score = _opt.length_score_unseen
        para = _opt.alpha_para_unseen
    elif _data == 1:
        min_score = _opt.min_score_seen
        max_score = _opt.max_score_seen
        mean_score = _opt.mean_score_seen
        length_score = _opt.length_score_seen
        para = _opt.alpha_para_seen
    elif _data == 2:
        min_score = _opt.min_score_train
        max_score = _opt.max_score_train
        mean_score = _opt.mean_score_train
        length_score = _opt.length_score_train
        para = _opt.alpha_para_train
    else:
        return 0

    mean_energy_score = _score.mean(0).item()
    
    if(mean_energy_score < min_score or mean_energy_score > max_score):
        alpha = 0
    else:
        if mean_energy_score > mean_score:
            alpha = (mean_energy_score - mean_score) / length_score
        else:
            alpha = (-mean_energy_score + mean_score) / length_score
        # alpha = (mean_energy_score - min_score) / length_score

    return para * alpha


def LearnSigmoid(_input, _weight):
    return 1/(1+torch.exp(-_weight * _input))

def get_seq_label(_energy_score, _offset, _weight, _is_max_min_probability = True):
    if _is_max_min_probability:
        max_score = _energy_score.max()
        min_score = _energy_score.min()
        OOD_confidence = (max_score - _energy_score) / (max_score - min_score)
        ID_confidence = 1 - OOD_confidence
        OOD_confidence = torch.reshape(OOD_confidence, (_energy_score.size(0), 1))
        ID_confidence = torch.reshape(ID_confidence, (_energy_score.size(0), 1))

        ID_confidence = torch.cat((ID_confidence, OOD_confidence), 1)

    else:
        # 
        energy_score = torch.reshape(_energy_score, (_energy_score.size(0), 1))
        energy_score = energy_score + _offset
        # ID_confidence = m(energy_score)
        # ID_confidence = torch.sigmoid(energy_score)
        ID_confidence = LearnSigmoid(energy_score, _weight)
        OOD_confidence = 1 - ID_confidence
        ID_confidence = torch.cat((ID_confidence, OOD_confidence), 1)
    return ID_confidence

def calc_ood_loss(feature, label, map_label, opt, data_train, netG, netMap, netF_CLS, netOODMap, netOOD, net_sigmoid, final_criterion, KL_Loss, bi_criterion):
    # 
    syn_feature, syn_label = generate_test(netG, data_train.unseenclasses, data_train.attribute, opt.ood_train_syn_num, opt)
    syn_map_label = map_label_all(syn_label, data_train.seenclasses, data_train.unseenclasses, data_train.ntrain_class)
    netF_CLS.zero_grad()
    netMap.zero_grad()
    net_sigmoid.zero_grad()

    if opt.cuda:
        syn_feature = syn_feature.cuda()
        syn_label = syn_label.cuda()
        syn_map_label = syn_map_label.cuda()
    start_select = syn_feature.size(0)
    syn_feature = torch.cat((syn_feature, feature), 0)
    syn_label = torch.cat((syn_label, label), 0)
    syn_map_label = torch.cat((syn_map_label, map_label), 0)
    end_select = syn_feature.size(0)

    # 
    embed = netMap(syn_feature)
    output, bi_feature, kl_input = netF_CLS(embed, data_train.seenclasses, data_train.unseenclasses)
    final_cls_loss  = final_criterion(output, syn_label)

    # ood
    embed_O = netOODMap(syn_feature)
    ood_output, logit_v = netOOD(embed_O)
    energy_score, _ = torch.max(ood_output, dim=1, keepdim=True)

    # kl loss
    # 
    indices = torch.arange(start_select, end_select).cuda()

    # softmax kl loss
    seen_kl_input = torch.index_select(kl_input, dim=0, index=indices)
    seen_logit_v = torch.index_select(logit_v, dim=0, index=indices)
    logits_distillation_loss = KL_Loss(F.log_softmax(seen_kl_input, dim=1), F.softmax(seen_logit_v, dim=1))

    # embed distillation loss
    embed_teacher = torch.index_select(embed_O, dim=0, index=indices)
    embed_student = torch.index_select(embed, dim=0, index=indices)
    # embed_distillation_loss = lrd_loss(embed_student, embed_teacher)

    label_embed_teacher = torch.index_select(syn_label, dim=0, index=indices)
    label_embed_student = torch.index_select(syn_label, dim=0, index=indices)
    embed_distillation_loss = batch_embed_distillation(embed_teacher, embed_student, label_embed_teacher, label_embed_student, bi_criterion)

    # kl_loss = KL_Loss(F.log_softmax(kl_input, dim=1), F.softmax(logit_v, dim=1))
    # sigmoid
    Sequence_label = net_sigmoid(energy_score)
    OOD_confidence = 1 - Sequence_label
    Sequence_label = torch.cat((Sequence_label, OOD_confidence), 1)
    OOD_contrastive_loss = batch_cosine_similarity(F.softmax(bi_feature, dim=1), F.softmax(Sequence_label, dim=1), syn_label, bi_criterion)
############################################
    # OOD_distillation_loss = fig_OOD_distillation_loss(bi_feature, Sequence_label)
    
############################################
    OOD_loss = OOD_contrastive_loss
    # OOD_loss = OOD_contrastive_loss + OOD_distillation_loss
    # distillation_loss = logits_distillation_loss
    distillation_loss = embed_distillation_loss + logits_distillation_loss
    return final_cls_loss, OOD_loss, distillation_loss

def fig_OOD_distillation_loss(_bi_feature, _Sequence_label):
    # KL
    KL_Loss = nn.KLDivLoss(reduction="batchmean")      
    loss = KL_Loss(F.log_softmax(_bi_feature, dim=1), F.softmax(_Sequence_label, dim=1))

    #
    # mse_loss = nn.MSELoss()
    # loss = mse_loss(_bi_feature, _Sequence_label)

    # 
    # l1_loss = nn.L1Loss()
    # loss = l1_loss(_bi_feature, _Sequence_label)

    # 
    # loss = js_divergence(_bi_feature, _Sequence_label)

    return loss

def kl_divergence(p, q):

    return (p * (p / q).log()).sum()

def js_divergence(p, q):

    p = F.softmax(p, dim=-1)  # 将张量转换为概率分布
    q = F.softmax(q, dim=-1)  # 将张量转换为概率分布
    m = (p + q) / 2
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2


def batch_embed_distillation(real_seen_feat, syn_seen_feat, real_seen_label, syn_seen_label, bi_criterion):
    a_number = real_seen_feat.size(0)
    b_number = syn_seen_feat.size(0)
    a_embedding = real_seen_feat.unsqueeze(1).repeat(1, b_number, 1).view(-1, real_seen_feat.size(1))
    b_embedding = syn_seen_feat.unsqueeze(0).repeat(a_number, 1, 1).view(-1, syn_seen_feat.size(1))

    similarity = (torch.cosine_similarity(a_embedding, b_embedding) + 1) / 2
    similarity = similarity.view(similarity.size(0), -1)

    real_seen_label = real_seen_label.contiguous().view(1, -1)
    syn_seen_label = syn_seen_label.contiguous().view(-1, 1)
    
    # 计算ground_truth_label
    ground_truth_label = torch.eq(real_seen_label, syn_seen_label).float().view(-1, 1)

    batch_embed_loss = bi_criterion(similarity, ground_truth_label)

    return batch_embed_loss

def batch_cosine_similarity(a,b,syn_label,bi_criterion):
    a_number = a.size(0)
    b_number = b.size(0)
    a_embedding = a.unsqueeze(1).repeat(1, b_number, 1).view(-1, a.size(1))
    b_embedding = b.unsqueeze(0).repeat(a_number, 1, 1).view(-1, b.size(1))
     # Compute cosine similarity and rescale it to the range [0, 1]
    similarity = (torch.cosine_similarity(a_embedding, b_embedding) + 1) / 2
    # similarity = torch.cosine_similarity(a_embedding, b_embedding)
    similarity = similarity.view(similarity.size(0), -1)
    syn_label = syn_label.contiguous().view(-1, 1)
    ground_truth_label = torch.eq(syn_label, syn_label.T).float().view(-1, 1)

    # Apply logit transformation to similarity
    # logits = torch.log(similarity / (1 - similarity))
    # OOD_contrastive_loss = bi_criterion(logits, ground_truth_label)
    OOD_contrastive_loss = bi_criterion(similarity, ground_truth_label)

    return OOD_contrastive_loss


def lrd_loss(qi, q_plus_i):
    # Calculate the dot product between qi and q_plus_i along the feature dimension (n)
    dot_product = torch.sum(qi * q_plus_i, dim=1)

    # Calculate the L2 norms of qi and q_plus_i along the feature dimension (n)
    qi_norm = torch.norm(qi, p=2, dim=1)
    q_plus_i_norm = torch.norm(q_plus_i, p=2, dim=1)

    # Calculate the LRD loss
    lrd = 1 - (dot_product / (qi_norm * q_plus_i_norm))
    
    # Optional: Average LRD loss over the batch (m)
    lrd_loss = torch.mean(lrd)

    return lrd_loss

def ground_truth(a):
    # a_size = a.size(0)
    # b = a.unsqueeze(0).repeat(a_size,1).view(-1,a_size*a_size).squeeze().cuda()
    # c = a.unsqueeze(1).repeat(1,a_size).view(-1,a_size*a_size).squeeze().cuda()
    # d = torch.Tensor([int(bb==cc) for bb,cc in zip(b,c)])
    # print(torch.sum(d))
    # return d
    num = a.size(0)
    real_label = torch.FloatTensor(a.size(0)*a.size(0))
    temp = torch.FloatTensor(a.size(0)).fill_(0)
    for i in range(num):
        idx = (a == a[i])
        idy = (a != a[i])
        temp[idx] = 1
        temp[idy] = 0
        real_label.narrow(0, i * num, num).copy_(temp)
    # print(torch.sum(real_label))
    real_label = real_label.view(real_label.size(0), -1)
    return real_label