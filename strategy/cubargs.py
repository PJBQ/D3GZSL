# from matplotlib.pyplot import cla
import torch
# import argparse

class init_args:
    def __init__(self):
        # dataset setting
        self.dataset = "cub"
        self.x_dim = 2048
        self.embedSize = 2048
        self.ood_embedSize = 1024
        self.attr_dim = 1024
        self.nz = 1024    # the same dimensionality as the class embedding
        self.n_class_all = 200
        self.n_class_seen = 150
        self.ce_syn_num = 100
        self.syn_num = 300
        self.ood_train_syn_num = 50
        self.batch_size = 2048   # for train
        self.test_batch_size = 256   # for test
        self.cls_batch_size = 256
        self.gan_batch_size = 128
        self.f_batch_size = 256
        self.out_dim = 200
        

        # share paremeter
        self.manualSeed = 3483
        self.cuda = True
        self.validation = False
        self.preprocessing = True
        self.standardization = False
        self.lr = 0.0001
        self.dependent_lr = 0.001
        self.lr_dec_rate = 0.99
        self.lr_decay_epoch = 100
        self.beta1 = 0.5
        self.lambda1 = 10
        self.gzsl = True

        # trick
        self.use_gan_ood = False

        self.use_cls_loss = True
        self.final_ood = True

        self.use_cls_ood = False
        self.pre_ood = False

        self.is_max_min_probability=False
        self.offset = 0
        self.sigmoid_weight = 1

        

        # epoch setting
        self.g_epoch = 2000
        self.c_epoch = 20
        self.f_epoch = 25
        self.o_epoch = 30
        self.critic_iter = 5
        self.start_ood = -1
        self.start_dependent_test = -1

        # ood setting
        self.id_threshold = -1.2
        self.ood_threshold = -1.2
        self.threshold = -1.2
        self.m_in = -23
        self.m_out = -5
        self.ood_loss_para = 0.01
        self.oodtrain_loss_para = 0.001
        self.oodclass_loss_para = 0.0001
        self.bi_loss_para = 0.0001

        self.kl_loss_para = 0.0001
        

        # ratio
        self.beta = 0.01
        self.lmbda = 10.0
        
        # path
        # self.res_path = './data/awa1/res101.mat'
        # self.att_path = './data/awa1/att_splits.mat'
        self.res_path = './data/cub/res101.mat'
        self.att_path = './data/cub/att_splits.mat'
        
        # paper acc
        self.paper_H_acc = 0.497
        self.paper_seen_acc = 0.577
        self.paper_unseen_acc = 0.437

        # acc method
        self.acc_class = False
        self.acc_top1 = False

        self.png_root = './img/'
        self.png_suffix = '.png'

        # test
        self.dependent_test = False

