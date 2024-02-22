from __future__ import print_function
import argparse
import sys
sys.path.append("..")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import util
import oodutil
import classifier_embed_contras
import model
import losses
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cubargs
import awaargs
import awa2args
import floargs
import sunargs
import wandb
my_opt = cubargs.init_args()
# 
model_folder = 'models'
model_filename = 'netG_weights.pth'
map_filename = 'Map_weights.pth'
netG_path = os.path.join(model_folder, my_opt.dataset, model_filename)
Map_path = os.path.join(model_folder, my_opt.dataset, map_filename)

wandb.init(project="adloss", entity="dearcat", name="cub_ce_strategy", config=my_opt)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='FLO')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='sent',help='att or sent')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', type=bool, default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', type=bool, default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=2048, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024 , help='size of semantic features')
parser.add_argument('--nz', type=int, default=1024, help='noise for generation')
parser.add_argument('--embedSize', type=int, default=2048, help='size of embedding h')
parser.add_argument('--outzSize', type=int, default=512, help='size of non-liner projection z')

## network architechure
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator G')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator D')
parser.add_argument('--nhF', type=int, default=2048, help='size of the hidden units comparator network F')

parser.add_argument('--ins_weight', type=float, default=0.001, help='weight of the classification loss when learning G')
parser.add_argument('--cls_weight', type=float, default=0.001, help='weight of the score function when learning G')
parser.add_argument('--ins_temp', type=float, default=0.1, help='temperature in instance-level supervision')
parser.add_argument('--cls_temp', type=float, default=0.1, help='temperature in class-level supervision')

parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to training')
parser.add_argument('--lr_decay_epoch', type=int, default=100, help='conduct learning rate decay after every 100 epochs')
parser.add_argument('--lr_dec_rate', type=float, default=0.99, help='learning rate decay rate')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=3483, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--nclass_seen', type=int, default=150, help='number of all classes')

parser.add_argument('--gpus', default='0', help='the number of the GPU to use')
opt = parser.parse_args()
print(opt)

# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt, is_train=True, is_seen=False)
loader_train = DataLoader(data, batch_size=my_opt.gan_batch_size, shuffle=True, drop_last=True)
print("# of training samples: ", data.ntrain)

netG = model.MLP_G(opt)
netMap = model.Embedding_Net(opt)
netD = model.MLP_CRITIC(opt)
F_ha = model.Dis_Embed_Att(opt)
netOODMap = model.Embedding_OOD_Net(my_opt)
netOOD = model.OOD_MLP_DEEP(my_opt)
netF_CLS = model.F_MLP(my_opt)

ood_criterion = nn.NLLLoss()   
KL_Loss = nn.KLDivLoss(reduction="batchmean")     
bi_criterion = nn.BCEWithLogitsLoss()       
final_criterion = nn.NLLLoss()     

net_sigmoid = model.LearnSigmoid(my_opt.sigmoid_weight, my_opt.offset)

model_path = './models/' + opt.dataset
if not os.path.exists(model_path):
    os.makedirs(model_path)

if len(opt.gpus.split(','))>1:
    netG=nn.DataParallel(netG)
    netD = nn.DataParallel(netD)
    netMap = nn.DataParallel(netMap)
    F_ha = nn.DataParallel(F_ha)


contras_criterion = losses.SupConLoss_clear(opt.ins_temp)

input_res = torch.FloatTensor(my_opt.batch_size, my_opt.x_dim)
input_att = torch.FloatTensor(my_opt.batch_size, my_opt.attr_dim)
noise_gen = torch.FloatTensor(my_opt.batch_size, my_opt.nz)
input_label = torch.LongTensor(my_opt.batch_size)
original_label = torch.LongTensor(my_opt.batch_size)

if opt.cuda:
    netG.cuda()
    netD.cuda()
    netMap.cuda()
    F_ha.cuda()
    netOOD.cuda()
    netF_CLS.cuda()
    netOODMap.cuda()

    net_sigmoid.cuda()
    final_criterion.cuda()
    ood_criterion.cuda()
    KL_Loss.cuda()
    bi_criterion.cuda()


    input_res = input_res.cuda()
    noise_gen, input_att = noise_gen.cuda(), input_att.cuda()
    input_label = input_label.cuda()
    original_label = original_label.cuda()


def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    original_label.copy_(batch_label)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))



def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
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


# setup optimizer
import itertools

optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(itertools.chain(netD.parameters(), netMap.parameters(), F_ha.parameters()), lr=opt.lr,
                        betas=(opt.beta1, 0.999))
optimizerOOD = optim.Adam(itertools.chain(netOOD.parameters(), netOODMap.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerFCLS = optim.Adam(netF_CLS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerSigmoid = optim.Adam(net_sigmoid.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

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

# use the for-loop to save the GPU-memory
def class_scores_for_loop(embed, input_label, relation_net):
    all_scores=torch.FloatTensor(embed.shape[0],opt.nclass_seen).cuda()
    for i, i_embed in enumerate(embed):
        expand_embed = i_embed.repeat(opt.nclass_seen, 1)#.reshape(embed.shape[0] * opt.nclass_seen, -1)
        all_scores[i]=(torch.div(relation_net(torch.cat((expand_embed, data.attribute_seen.cuda()), dim=1)),opt.cls_temp).squeeze())
    score_max, _ = torch.max(all_scores, dim=1, keepdim=True)
    # normalize the scores for stable training
    scores_norm = all_scores - score_max.detach()
    mask = F.one_hot(input_label, num_classes=opt.nclass_seen).float().cuda()
    exp_scores = torch.exp(scores_norm)
    log_scores = scores_norm - torch.log(exp_scores.sum(1, keepdim=True))
    cls_loss = -((mask * log_scores).sum(1) / mask.sum(1)).mean()
    return cls_loss

# It is much faster to use the matrix, but it cost much GPU memory.
def class_scores_in_matrix(embed, input_label, relation_net):
    expand_embed = embed.unsqueeze(dim=1).repeat(1, opt.nclass_seen, 1).reshape(embed.shape[0] * opt.nclass_seen, -1)
    expand_att = data.attribute_seen.unsqueeze(dim=0).repeat(embed.shape[0], 1, 1).reshape(
        embed.shape[0] * opt.nclass_seen, -1).cuda()
    all_scores = torch.div(relation_net(torch.cat((expand_embed, expand_att), dim=1)),opt.cls_temp).reshape(embed.shape[0],
                                                                                                    opt.nclass_seen)
    score_max, _ = torch.max(all_scores, dim=1, keepdim=True)
    scores_norm = all_scores - score_max.detach()
    mask = F.one_hot(input_label, num_classes=opt.nclass_seen).float().cuda()
    exp_scores = torch.exp(scores_norm)
    log_scores = scores_norm - torch.log(exp_scores.sum(1, keepdim=True))
    cls_loss = -((mask * log_scores).sum(1) / mask.sum(1)).mean()
    return cls_loss


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

#################

for epoch in range(my_opt.o_epoch):
# for epoch in range(0,200):
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
        class_loss = my_opt.oodtrain_loss_para*ood_criterion(output, map_label_O)

        loss = class_loss
        loss_sum = loss_sum + loss
        loss.backward()
        optimizerOOD.step()
    
    acc_cls = oodutil.val_ood_cls(data.test_seen_feature, data.map_test_seen_label, netOOD, netOODMap, my_opt, _start=0, _end=my_opt.n_class_seen)
    acc_id = oodutil.val_ood(data.test_seen_feature, data.map_test_seen_label, netOOD, netOODMap, my_opt, is_ood=False)
    acc_ood = oodutil.val_ood(data.test_unseen_feature, data.map_test_unseen_label, netOOD, netOODMap, my_opt, is_ood=True)

    wandb.log({
        'Loss_ood':loss_sum,
        'acc_oodcls':acc_cls,
        'acc_id':acc_id,
        'acc_ood':acc_ood
    })
    print('[%d/%d] Loss_ood: %.4f ## cls acc: %.4f ## id acc: %.4f ## ood acc: %.4f' % (epoch + 1, my_opt.o_epoch, loss_sum, acc_cls, acc_id, acc_ood))

netOOD.eval()
for p in netOOD.parameters():  # reset requires_grad
    p.requires_grad = False      
netOODMap.eval()
for p in netOODMap.parameters():  # reset requires_grad
    p.requires_grad = False 

#################
for epoch in range(my_opt.g_epoch):
    FP = 0
    mean_lossD = 0
    mean_lossG = 0
    sum_lossG = 0
    sum_lossFCLS = 0
    for i in range(0, data.ntrain, my_opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in netMap.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in F_ha.parameters():  # reset requires_grad
            p.requires_grad = True
        
        for p in netG.parameters():  # reset requires_grad
            p.requires_grad = False

        for iter_d in range(opt.critic_iter):
            sample()    
            netD.zero_grad()
            netMap.zero_grad()
            #
            # train with realG
            # sample a mini-batch
            sparse_real = opt.resSize - input_res[1].gt(0).sum()
            embed_real, outz_real = netMap(input_res)
            criticD_real = netD(input_res, input_att)
            criticD_real = criticD_real.mean()

            # CONTRASITVE LOSS
            real_ins_contras_loss = contras_criterion(outz_real, input_label)

            # train with fakeG
            noise_gen.normal_(0, 1)
            fake = netG(noise_gen, input_att)
            fake_norm = fake.data[0].norm()         
            sparse_fake = fake.data[0].eq(0).sum()  
            criticD_fake = netD(fake.detach(), input_att)
            criticD_fake = criticD_fake.mean()

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            Wasserstein_D = criticD_real - criticD_fake     

            cls_loss_real = class_scores_for_loop(embed_real, input_label, F_ha)

            D_cost = criticD_fake - criticD_real + gradient_penalty + real_ins_contras_loss + cls_loss_real

            D_cost.backward()
            optimizerD.step()

            wandb.log({
                'D_loss':D_cost,
                # 'weight_loss_D':weight_loss_D
            })

        # (2) Update G network: optimize WGAN-GP objective, Equation (2)

        for p in netD.parameters(): 
            p.requires_grad = False  
        for p in netMap.parameters(): 
            p.requires_grad = False
        for p in F_ha.parameters(): 
            p.requires_grad = False

        for p in netG.parameters(): 
            p.requires_grad = True

        netG.zero_grad()
        noise_gen.normal_(0, 1)
        fake = netG(noise_gen, input_att)

        embed_fake, outz_fake = netMap(fake)

        criticG_fake = netD(fake, input_att)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake

        embed_real, outz_real = netMap(input_res)

        all_outz = torch.cat((outz_fake, outz_real.detach()), dim=0)

        fake_ins_contras_loss = contras_criterion(all_outz, torch.cat((input_label, input_label), dim=0))

        cls_loss_fake = class_scores_for_loop(embed_fake, input_label, F_ha)

        errG = G_cost + opt.ins_weight * fake_ins_contras_loss + opt.cls_weight * cls_loss_fake  # + opt.ins_weight * c_errG

        sum_lossG += errG
        errG.backward()
        optimizerG.step()

        wandb.log({
            'G_loss':errG,
            # 'weight_loss_G':weight_loss_G
        })


    F_ha.zero_grad()
    if (epoch + 1) % opt.lr_decay_epoch == 0:
        for param_group in optimizerD.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
        for param_group in optimizerG.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_dec_rate

    if (epoch + 1) % opt.lr_decay_epoch == 0:
        for param_group in optimizerFCLS.param_groups:
            param_group['lr'] = param_group['lr'] * 0.99

    mean_lossG /= data.ntrain / opt.batch_size
    mean_lossD /= data.ntrain / opt.batch_size

    print(
        '[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, real_ins_contras_loss:%.4f, fake_ins_contras_loss:%.4f, cls_loss_real: %.4f, cls_loss_fake: %.4f'
        % (epoch, opt.nepoch, D_cost, G_cost, Wasserstein_D, real_ins_contras_loss, fake_ins_contras_loss, cls_loss_real, cls_loss_fake))

    # evaluate the model, set G to evaluation mode
    # oodutil.print_net_para(net_sigmoid)
    netG.eval()
    netF_CLS.eval()
    for p in netMap.parameters():  # reset requires_grad
        p.requires_grad = False
    for p in netF_CLS.parameters():  # reset requires_grad
        p.requires_grad = False  # avoid computation

    if my_opt.dependent_test:
        with torch.no_grad():
            acc_seen = oodutil.val_gzsl(data.test_seen_feature, data.test_seen_label, data.seenclasses, data.unseenclasses, netMap, netF_CLS, my_opt, _is_seen=True)
            acc_unseen = oodutil.val_gzsl(data.test_unseen_feature, data.test_unseen_label, data.seenclasses, data.unseenclasses, netMap, netF_CLS, my_opt, _is_seen=False)
            print('[%d/%d] ### seen acc: %.4f  unseen acc: %.4f' % (epoch + 1, my_opt.g_epoch, acc_seen, acc_unseen))
            best_seen, best_unseen, best_H = oodutil.get_best_acc(acc_seen, acc_unseen, best_seen, best_unseen,best_H)
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

            wandb.log({
                'acc_seen':acc_seen,
                'acc_unseen':acc_unseen,
                'best_H_seen':best_H_seen,
                'best_H_unseen':best_H_unseen,
                'best_H_H':best_H_H
            })
            print('*************** H      *****************unseen=%.4f, seen=%.4f, h=%.4f, epoch=%d' % (best_H_unseen, best_H_seen, best_H_H, best_H_epoch))
            print('*************** seen   *****************unseen=%.4f, seen=%.4f, h=%.4f, epoch=%d' % (best_seen_unseen, best_seen_seen, best_seen_H, best_seen_epoch))
            print('*************** unseen *****************unseen=%.4f, seen=%.4f, h=%.4f, epoch=%d' % (best_unseen_unseen, best_unseen_seen, best_unseen_H, best_unseen_epoch))
    else:
        if opt.gzsl: # Generalized zero-shot learning
            test_syn_feature, test_syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

            train_X = torch.cat((data.train_feature, test_syn_feature), 0)
            train_Y = torch.cat((data.train_label, test_syn_label), 0)
            train_Y_map = torch.cat((data.train_label, test_syn_label), 0)

            # 
            train_Y_map.copy_(util.map_label_all(train_Y, data.seenclasses, data.unseenclasses, data.ntrain_class))


            nclass = opt.nclass_all
            cls = classifier_embed_contras.CLASSIFIER(my_opt, train_X, train_Y, train_Y_map, netMap, opt.embedSize, data, nclass, opt.cuda,
                                                    my_opt.dependent_lr, 0.5, my_opt.f_epoch, my_opt.ce_syn_num,
                                                    True)
            
            print('[%d/%d] ### unseen=%.4f, seen=%.4f, h=%.4f' % (epoch + 1, my_opt.g_epoch, cls.acc_unseen, cls.acc_seen, cls.H))
            print('bi_ood_model_acc=%.4f, bi_model_acc=%.4f' % (cls.best_bi_ood_model_acc, cls.best_bi_model_acc))

            if best_H_H < cls.H:
                best_H_H = cls.H
                best_H_seen = cls.acc_seen
                best_H_unseen = cls.acc_unseen
                best_H_epoch = epoch
                # 
                torch.save(netG.state_dict(), netG_path)
                torch.save(netMap.state_dict(), Map_path)
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
            wandb.log({
                'acc_seen':cls.acc_seen,
                'acc_unseen':cls.acc_unseen,
                'best_H_seen':best_H_seen,
                'best_H_unseen':best_H_unseen,
                'best_H_H':best_H_H
            })
            print('*************** H      *****************unseen=%.4f, seen=%.4f, h=%.4f, epoch=%d' % (best_H_unseen, best_H_seen, best_H_H, best_H_epoch))
            print('*************** seen   *****************unseen=%.4f, seen=%.4f, h=%.4f, epoch=%d' % (best_seen_unseen, best_seen_seen, best_seen_H, best_seen_epoch))
            print('*************** unseen *****************unseen=%.4f, seen=%.4f, h=%.4f, epoch=%d' % (best_unseen_unseen, best_unseen_seen, best_unseen_H, best_unseen_epoch))

        else:  # conventional zero-shot learning
            syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
            cls = classifier_embed_contras.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), netMap,
                                                    opt.embedSize, data,
                                                    data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 100,
                                                    opt.syn_num,
                                                    False)
            acc = cls.acc
            print('unseen class accuracy=%.4f '%acc)


    # reset G to training mode
    netG.train()
    netF_CLS.train()

