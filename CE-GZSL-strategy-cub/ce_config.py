from easydict import EasyDict

ce = EasyDict()


ce.log_path = 'logs_test'
ce.is_ood = True

# ce.o_epoch = 20

# ood para
ce.m_in = -23
ce.m_out = -5
ce.threshold = -16
ce.syn_num = 100


# epoch
ce.epoch_ood = 15
ce.epoch_cl = 25

# loss 
ce.ood_loss_para = 0.1
ce.bi_loss_para = 1

# 
ce.use_gan_ood = False
ce.start_ood = 100
ce.ood_train_syn_num = 10
