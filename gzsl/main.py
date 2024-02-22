import os

from pyexpat import features

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random

import torch
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import argsawa
import argsawa2
import argscub
import argsflo
import argssun
import model
import util
import val_test
import wandb

opt = argscub.init_args()
wandb.init(project="addloss", entity="dearcat", name="cub_batch_embed_loss_cls", config=opt)

# set manualSeed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# TODO
cudnn.benchmark = True

# set cuda
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data_train = util.DATA_LOADER(opt, is_train=True,is_seen=False,is_syn=False,syn_feature=None,syn_label=None)
loader_train = DataLoader(data_train, batch_size=opt.gan_batch_size, shuffle=True, drop_last=True)
print("# of training samples: ", data_train.ntrain)

# model
netG = model.Generator(opt)
netD = model.Discriminator(opt)
netOOD = model.OOD_MLP_DEEP(opt)
netF_CLS = model.F_MLP(opt)
netMap = model.Embedding_Net(opt)
netOODMap = model.Embedding_Net(opt)

ood_criterion = nn.NLLLoss()    # 
KL_Loss = nn.KLDivLoss(reduction="batchmean")       # 
bi_criterion = nn.BCEWithLogitsLoss()       # 
final_criterion = nn.NLLLoss()      # 

net_sigmoid = model.LearnSigmoid(opt.sigmoid_weight, opt.offset)

model_path = './models/' + opt.dataset
if not os.path.exists(model_path):
    os.makedirs(model_path)

# if len(opt.gpus.split(','))>1:
#     netG=nn.DataParallel(netG)

# feature
features = torch.FloatTensor(opt.batch_size, opt.x_dim)
input_att = torch.FloatTensor(opt.batch_size, opt.attr_dim)
noise_gen = torch.FloatTensor(opt.batch_size, opt.nz)
label = torch.LongTensor(opt.batch_size)
map_label = torch.LongTensor(opt.batch_size)

if opt.cuda:
    netG.cuda()
    netD.cuda()
    netOOD.cuda()
    net_sigmoid.cuda()
    netMap.cuda()
    netOODMap.cuda()
    final_criterion.cuda()
    ood_criterion.cuda()
    netF_CLS.cuda()
    KL_Loss.cuda()
    bi_criterion.cuda()

    features = features.cuda()
    noise_gen = noise_gen.cuda()
    input_att = input_att.cuda()
    label = label.cuda()
    map_label = map_label.cuda()

# setup optimizer
import itertools

optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerOOD = optim.Adam(itertools.chain(netOOD.parameters(), netOODMap.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerFCLS = optim.Adam(netF_CLS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerSigmoid = optim.Adam(net_sigmoid.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerMap = optim.Adam(netMap.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def sample():
    batch_feature, batch_label, batch_att, batch_map_label = data_train.next_batch(opt.batch_size)
    features.copy_(batch_feature)
    input_att.copy_(batch_att)
    label.copy_(batch_label)
    map_label.copy_(batch_map_label)


def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.x_dim)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attr_dim)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(syn_noise, syn_att)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    # print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, input_att)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


best_H_H = 0
best_H_seen = 0
best_H_unseen = 0
best_seen_H = 0
best_seen_seen = 0
best_seen_unseen = 0
best_unseen_H = 0
best_unseen_seen = 0 
best_unseen_unseen = 0
best_H_epoch = 0
best_seen_epoch = 0
best_unseen_epoch = 0

best_H = 0
best_seen = 0
best_unseen = 0
# train
# energy_score_b = util.get_energy_score(data_train.test_seen_feature, data_train.map_test_seen_label, netOOD, opt)
for epoch in range(opt.o_epoch):
    loss_sum = 0.0
    acc_cls = 0
    acc_id = 0
    acc_ood = 0
    for feature_O, label_O, att_O, map_label_O in loader_train:
        if opt.cuda:
            feature_O = feature_O.cuda()
            label_O = label_O.cuda()
            map_label_O = map_label_O.cuda()
        optimizerOOD.zero_grad()
        embed = netOODMap(feature_O)
        output, logit_v = netOOD(embed)     
        class_loss = opt.oodtrain_loss_para*ood_criterion(output, map_label_O)

        loss = class_loss
        loss_sum = loss_sum + loss
        loss.backward()
        optimizerOOD.step()
    
    acc_cls = util.val_ood_cls(data_train.test_seen_feature, data_train.map_test_seen_label, netOOD, netOODMap, opt, _start=0, _end=opt.n_class_seen)
    acc_id = util.val_ood(data_train.test_seen_feature, data_train.map_test_seen_label, netOOD, netOODMap, opt, is_ood=False)
    acc_ood = util.val_ood(data_train.test_unseen_feature, data_train.map_test_unseen_label, netOOD, netOODMap, opt, is_ood=True)

    wandb.log({
        'Loss_ood':loss_sum,
        'acc_oodcls':acc_cls,
        'acc_id':acc_id,
        'acc_ood':acc_ood
    })
    print('[%d/%d] Loss_ood: %.4f ## cls acc: %.4f ## id acc: %.4f ## ood acc: %.4f' % (epoch + 1, opt.o_epoch, loss_sum, acc_cls, acc_id, acc_ood))


netOOD.eval()
for p in netOOD.parameters():  # reset requires_grad
    p.requires_grad = False      
netOODMap.eval()
for p in netOODMap.parameters():  # reset requires_grad
    p.requires_grad = False 

for epoch in range(opt.g_epoch):
    FP = 0
    sum_lossD = 0
    sum_lossG = 0
    sum_lossFCLS = 0
    sum_lossFOOD = 0

    # for feature, label, att, map_label in loader_train:
    for i in range(0, data_train.ntrain, opt.batch_size):

        netD.train()
        netG.train()
        netMap.train()
        netF_CLS.train()
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in netMap.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in netG.parameters():  # reset requires_grad
            p.requires_grad = False

        for p in netF_CLS.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in net_sigmoid.parameters():  # reset requires_grad
            p.requires_grad = True

        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()
            # train with real
            criticD_real = netD(features, input_att)
            criticD_real = criticD_real.mean()

            # train with fake
            noise_gen.normal_(0, 1)
            fake = netG(noise_gen, input_att)

            criticD_fake = netD(fake.detach(), input_att)     # fake.detach : not update netG
            criticD_fake = criticD_fake.mean()

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, features, fake.data, input_att)
            Wasserstein_D = criticD_real - criticD_fake

            # loss
            D_loss = criticD_fake - criticD_real + gradient_penalty

            # 
            if opt.final_ood and epoch%1 == 0 and epoch > -1:
                final_cls_loss,sequence_loss,kl_loss = util.calc_ood_loss(features, label, map_label, opt, data_train, netG, netMap, netF_CLS, netOODMap, netOOD, net_sigmoid,final_criterion, KL_Loss,bi_criterion)

                weight_loss_D = opt.oodclass_loss_para * final_cls_loss + opt.bi_loss_para * sequence_loss + opt.kl_loss_para * kl_loss

                D_loss += weight_loss_D
            
            sum_lossD += D_loss
            D_loss.backward()
            optimizerD.step()
            optimizerMap.step()
            optimizerFCLS.step()
            optimizerSigmoid.step()
            
            wandb.log({
                'D_loss':D_loss,
                'weight_loss_D':weight_loss_D
            })
            

        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation
        for p in netF_CLS.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in netMap.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in net_sigmoid.parameters():  # reset requires_grad
            p.requires_grad = True

        for p in netG.parameters():  # reset requires_grad
            p.requires_grad = True
        
        
        netG.zero_grad()
        netMap.zero_grad()
        netF_CLS.zero_grad()
        noise_gen.normal_(0, 1)
        fake = netG(noise_gen, input_att)

        criticG_fake = netD(fake, input_att)
        criticG_fake = criticG_fake.mean()

        # loss
        G_loss = -criticG_fake

        # print('[%d/%d] Loss_Gan_batch: %.4f' % (epoch + 1, opt.g_epoch, G_loss))  

        if opt.final_ood and epoch%1 == 0 and epoch > -1:
            final_cls_loss,sequence_loss,kl_loss = util.calc_ood_loss(features, label, map_label, opt, data_train, netG, netMap, netF_CLS, netOODMap, netOOD, net_sigmoid,final_criterion, KL_Loss,bi_criterion)

            weight_loss_G = opt.oodclass_loss_para * final_cls_loss + opt.bi_loss_para * sequence_loss + opt.kl_loss_para * kl_loss

                
            sum_lossFCLS += weight_loss_G
            G_loss += weight_loss_G

        sum_lossG += G_loss
        G_loss.backward()
        optimizerG.step()
        optimizerMap.step()
        optimizerFCLS.step()
        optimizerSigmoid.step()

        wandb.log({
            'G_loss':G_loss,
            'weight_loss_G':weight_loss_G
        })


    # adjust learning rate
    if (epoch + 1) % 20 == 0:
        for param_group in optimizerD.param_groups:
            param_group['lr'] = param_group['lr'] * 1.09
        for param_group in optimizerG.param_groups:
            param_group['lr'] = param_group['lr'] * 1.09
    if (epoch + 1) % 20 == 0:
        for param_group in optimizerMap.param_groups:
            param_group['lr'] = param_group['lr'] * 0.98
        for param_group in optimizerFCLS.param_groups:
            param_group['lr'] = param_group['lr'] * 0.98
        for param_group in optimizerSigmoid.param_groups:
            param_group['lr'] = param_group['lr'] * 0.99

    # mean_lossG /= data_train.ntrain / opt.batch_size
    # mean_lossD /= data_train.ntrain / opt.batch_size
    print(
        '[%d/%d] Loss_D: %.4f Loss_G: %.4f, Loss_FCLS: %.4f, Loss_FOOD: %.4f'
        % (epoch, opt.g_epoch, sum_lossD, sum_lossG, sum_lossFCLS, sum_lossFOOD))

    netG.eval()
    netMap.eval()
    netF_CLS.eval()

    for p in netG.parameters():  # reset requires_grad
        p.requires_grad = False  # avoid computation
    for p in netMap.parameters():  # reset requires_grad
        p.requires_grad = False  # avoid computation
    for p in netF_CLS.parameters():  # reset requires_grad
        p.requires_grad = False  # avoid computation



    # val
    if opt.dependent_test:
        with torch.no_grad():
            acc_seen = util.val_gzsl(data_train.test_seen_feature, data_train.test_seen_label, data_train.seenclasses, data_train.unseenclasses, netMap, netF_CLS, opt, _is_seen=True)
            acc_unseen = util.val_gzsl(data_train.test_unseen_feature, data_train.test_unseen_label, data_train.seenclasses, data_train.unseenclasses, netMap, netF_CLS, opt, _is_seen=False)
            print('[%d/%d] ### seen acc: %.4f  unseen acc: %.4f' % (epoch + 1, opt.g_epoch, acc_seen, acc_unseen))
            best_seen, best_unseen, best_H = util.get_best_acc(acc_seen, acc_unseen, best_seen, best_unseen,best_H)
            if best_H_H < best_H:
                best_H_H = best_H
                best_H_seen = best_seen
                best_H_unseen = best_unseen       
                best_H_epoch = epoch
            if best_seen_seen < best_seen:
                best_seen_H = best_H
                best_seen_seen = best_seen
                best_seen_unseen = best_unseen
                best_seen_epoch = epoch
            if best_unseen_unseen < best_unseen:
                best_unseen_H = best_H
                best_unseen_seen = best_seen
                best_unseen_unseen = best_unseen
                best_unseen_epoch = epoch
            print('*************** H      *****************unseen=%.4f, seen=%.4f, h=%.4f, epoch=%d' % (best_H_unseen, best_H_seen, best_H_H, best_H_epoch))
            print('*************** seen   *****************unseen=%.4f, seen=%.4f, h=%.4f, epoch=%d' % (best_seen_unseen, best_seen_seen, best_seen_H, best_seen_epoch))
            print('*************** unseen *****************unseen=%.4f, seen=%.4f, h=%.4f, epoch=%d' % (best_unseen_unseen, best_unseen_seen, best_unseen_H, best_unseen_epoch))
            wandb.log({
                'best_H_unseen':best_H_unseen,
                'best_H_seen':best_H_seen,
                'best_H_H':best_H_H,
                'unseen':acc_unseen,
                'seen':acc_seen,
                'epoch':epoch
            })
    else:
        syn_feature, syn_label = generate_syn_feature(netG, data_train.unseenclasses, data_train.attribute, opt.syn_num)
        # load data
        data_final = util.DATA_LOADER(opt, is_train=True,is_seen=False,is_syn=True,syn_feature=syn_feature,syn_label=syn_label)
        loader_final = DataLoader(data_final, batch_size=opt.f_batch_size, shuffle=True)
        # print("# of final training samples: ", data_final.ntrain)
        cls = val_test.CLASSIFIER(data_final, loader_final, opt, True, netMap, netOOD)
        print('[%d/%d]  unseen=%.4f, seen=%.4f, h=%.4f' % (epoch, opt.g_epoch, cls.acc_unseen, cls.acc_seen, cls.H))
        print('[%d/%d]  bi_ood_model_acc=%.4f, bi_model_acc=%.4f' % (epoch, opt.g_epoch, cls.best_bi_ood_model_acc, cls.best_bi_model_acc))

        if best_H_H < cls.H:
            best_H_H = cls.H
            best_H_seen = cls.acc_seen
            best_H_unseen = cls.acc_unseen
            best_H_epoch = epoch
        if best_seen_seen < cls.acc_seen:
            best_seen_H = cls.H
            best_seen_seen = cls.acc_seen
            best_seen_unseen = cls.acc_unseen
            best_seen_epoch = epoch
        if best_unseen_unseen < cls.acc_unseen:
            best_unseen_H = cls.H
            best_unseen_seen = cls.acc_seen
            best_unseen_unseen = cls.acc_unseen
            best_unseen_epoch = epoch
        print('*************** H      *****************unseen=%.4f, seen=%.4f, h=%.4f, epoch=%d' % (best_H_unseen, best_H_seen, best_H_H, best_H_epoch))
        print('*************** seen   *****************unseen=%.4f, seen=%.4f, h=%.4f, epoch=%d' % (best_seen_unseen, best_seen_seen, best_seen_H, best_seen_epoch))
        print('*************** unseen *****************unseen=%.4f, seen=%.4f, h=%.4f, epoch=%d' % (best_unseen_unseen, best_unseen_seen, best_unseen_H, best_unseen_epoch))
        wandb.log({
            'best_H_unseen':best_H_unseen,
            'best_H_seen':best_H_seen,
            'best_H_H':best_H_H,
            'unseen':cls.acc_unseen,
            'seen':cls.acc_seen,
            'epoch':epoch
        })
    netG.train()
    netMap.train()
    netF_CLS.train()