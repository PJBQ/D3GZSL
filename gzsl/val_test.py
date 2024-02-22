import torch
import model
import util
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CLASSIFIER:
    def __init__(self, _dataset, _loader, _opt, _generalized=True, _pre_netMap=None, _pre_OODNet=None):
        self.data = _dataset
        self.loader = _loader
        self.opt = _opt
        self.pre_OODNet = _pre_OODNet
        self.netMap = _pre_netMap
        self.model = F_MLP(self.opt)
        self.ood_model = OOD_MLP(self.opt)
        self.model.apply(util.weights_init)
        self.ood_model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()
        self.bi_CrossEntropyLoss = torch.nn.CrossEntropyLoss()

        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.opt.dependent_lr, betas=(self.opt.beta1, 0.999))
        self.ood_optimizer = optim.Adam(self.ood_model.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

        if self.opt.cuda:
            self.model.cuda()
            self.ood_model.cuda()
            self.pre_OODNet.cuda()
            self.criterion.cuda()
            self.bi_CrossEntropyLoss = self.bi_CrossEntropyLoss.cuda()

        if _generalized:
            if self.opt.use_cls_ood:
                if self.opt.pre_ood:
                    self.acc_seen, self.acc_unseen, self.H, self.best_bi_ood_model_acc, self.best_bi_model_acc = self.fit_classifier()
                else:
                    self.fit_train_ood()
                    self.acc_seen, self.acc_unseen, self.H, self.best_bi_ood_model_acc, self.best_bi_model_acc = self.fit_classifier()
            else:
                self.acc_seen, self.acc_unseen, self.H, self.best_bi_ood_model_acc, self.best_bi_model_acc = self.fit() 

        else:
            self.acc = self.fit_zsl()

    def fit_train_ood(self):
        
        for epoch in range(self.opt.o_epoch):
            loss_sum = 0
            self.ood_model.train()

            for p in self.ood_model.parameters():  # reset requires_grad
                p.requires_grad = True
            for feature, label, att, map_label in self.loader:
                if self.opt.cuda:
                    feature = feature.cuda()
                    label = label.cuda()
                    map_label = map_label.cuda()
                self.ood_optimizer.zero_grad()
                output, logit_v = self.ood_model(feature)

                class_loss, ood_loss = util.fig_ood_loss(map_label, logit_v, output, self.criterion, self.opt)
                # print('class_loss : {:.4f}  ood_loss : {:.4f}'.format(class_loss.data, ood_loss.data))
                # loss = class_loss + ood_loss
                loss = class_loss
                loss.backward()
                self.ood_optimizer.step()
                loss_sum += loss

            # val
            self.ood_model.eval()
            for p in self.ood_model.parameters():  # reset requires_grad
                p.requires_grad = False
            acc_cls = util.val_ood_cls(self.data.test_seen_feature, self.data.map_test_seen_label, self.ood_model, self.opt, _start=0, _end=self.opt.n_class_seen)
            acc_id = util.val_ood(self.data.test_seen_feature, self.data.map_test_seen_label, self.ood_model, self.opt, is_ood=False)
            acc_ood = util.val_ood(self.data.test_unseen_feature, self.data.map_test_unseen_label, self.ood_model, self.opt, is_ood=True)
            acc_train_data = util.val_ood(self.data.train_feature, self.data.map_label, self.ood_model, self.opt, is_ood=False)
            # print('[%d/%d] Loss_ood: %.4f ## cls acc: %.4f ## train acc: %.4f ## id acc: %.4f ## ood acc: %.4f' % (epoch + 1, self.opt.o_epoch, loss_sum, acc_cls, acc_train_data, acc_id, acc_ood))
            
            
                 
    def fit_classifier(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        best_bi_ood_model_acc = 0
        best_bi_model_acc = 0

        if self.opt.pre_ood:
            oodnet = self.pre_OODNet
        else:
            oodnet = self.ood_model
        oodnet.eval()
        for p in oodnet.parameters():  # reset requires_grad
            p.requires_grad = False

        for epoch in range(self.opt.f_epoch):
            bi_model_sum_correct_sample = 0
            bi_model_sum_sample = 0
            bi_ood_model_sum_correct_sample = 0
            bi_ood_model_sum_sample = 0
            loss_sum = 0.0
            for feature, label, att, map_label in self.loader:
                self.optimizer.zero_grad()
                if self.opt.cuda:
                    feature = feature.cuda()
                    label = label.cuda()
                    map_label = map_label.cuda()

                output, bi_feature = self.model(feature, self.data.seenclasses, self.data.unseenclasses)
                loss  = self.criterion(output, label)
                ##########################################################################################
                # 
                _, logit_v = oodnet(feature)
                energy_score = -torch.logsumexp(logit_v, dim=1)
                threshold = self.opt.threshold
                di_label = torch.LongTensor(energy_score.size()).cuda()
                in_idx = (energy_score < threshold)
                out_idx = (energy_score >= threshold)
                di_label[in_idx] = 0
                di_label[out_idx] = 1

                # bi loss
                # 
                bi_loss = self.opt.bi_loss_para * self.bi_CrossEntropyLoss(bi_feature,di_label) 
                # print('loss : {:.4f}  bi_loss : {:.4f}'.format(loss.data, bi_loss.data))
                ##########################################################################################
                loss += bi_loss
                loss.backward()
                self.optimizer.step()
                loss_sum += loss

                ##########################################################################################
                # ood_model
                bi_ood_model_correct_sample, bi_ood_model_samples = self.bi_acc(di_label, map_label)
                bi_ood_model_sum_correct_sample += bi_ood_model_correct_sample
                bi_ood_model_sum_sample += bi_ood_model_samples
                # model
                _, pre_di_label = torch.max(bi_feature, 1)
                bi_model_correct_sample, bi_model_samples = self.bi_acc(pre_di_label, map_label)
                bi_model_sum_correct_sample += bi_model_correct_sample
                bi_model_sum_sample += bi_model_samples
                # print('val binary acc model ***: {:.4f}  correct_sample: {:.4f}  sum_sample: {:.4f}'.format(float(bi_model_correct_sample)/float(bi_model_samples), bi_model_correct_sample, bi_model_samples))

            # ood_model
            acc_bi_ood_model = float(bi_ood_model_sum_correct_sample) / float(bi_ood_model_sum_sample)
            # print('val binary acc ood model ***: {:.4f}  correct_sample: {:.4f}  sum_sample: {:.4f}'.format(acc_bi_ood_model, bi_ood_model_sum_correct_sample, bi_ood_model_sum_sample))
            
            # model 
            acc_bi_model = float(bi_model_sum_correct_sample) / float(bi_model_sum_sample)
            # print('val binary acc model     ###: {:.4f}  correct_sample: {:.4f}  sum_sample: {:.4f}'.format(acc_bi_model, bi_model_sum_correct_sample, bi_model_sum_sample))
            
            best_bi_ood_model_acc = acc_bi_ood_model if acc_bi_ood_model > best_bi_ood_model_acc else best_bi_ood_model_acc
            best_bi_model_acc = acc_bi_model if acc_bi_model > best_bi_model_acc else best_bi_model_acc
############################################################################################################################################

            acc_seen = self.val_gzsl(self.data.test_seen_feature, self.data.test_seen_label, self.data.seenclasses)
            acc_unseen = self.val_gzsl(self.data.test_unseen_feature, self.data.test_unseen_label, self.data.unseenclasses)
            print('[%d/%d] Loss_final: %.4f ### seen acc: %.4f  unseen acc: %.4f' % (epoch + 1, self.opt.f_epoch, loss_sum, acc_seen, acc_unseen))
            best_seen, best_unseen, best_H = self.get_best_acc(acc_seen, acc_unseen, best_seen, best_unseen,best_H)
        return best_seen, best_unseen, best_H, best_bi_ood_model_acc, best_bi_model_acc

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        best_bi_ood_model_acc = 0
        best_bi_model_acc = 0
        for epoch in range(self.opt.f_epoch):
            loss_sum = 0.0
            self.model.train()
            for feature, label, att, _ in self.loader:
                if self.opt.cuda:
                    feature = feature.cuda()
                    label = label.cuda()
                      
                self.optimizer.zero_grad()
                embed = self.netMap(feature)
                cls_pred = self.model(embed)
                loss = self.criterion(cls_pred, label)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss 

            self.model.eval()
            acc_seen = self.val_gzsl(self.data.test_seen_feature, self.data.test_seen_label, self.data.seenclasses)
            acc_unseen = self.val_gzsl(self.data.test_unseen_feature, self.data.test_unseen_label, self.data.unseenclasses)
            print('[%d/%d] Loss_final: %.4f ### seen acc: %.4f  unseen acc: %.4f' % (epoch + 1, self.opt.f_epoch, loss_sum, acc_seen, acc_unseen))
            best_seen, best_unseen, best_H = self.get_best_acc(acc_seen, acc_unseen, best_seen, best_unseen,best_H)
        return best_seen, best_unseen, best_H, best_bi_ood_model_acc, best_bi_model_acc

    def bi_acc(self, pre_label, map_label):
        real_label = torch.LongTensor(len(map_label)).cuda()
        real_label.copy_(map_label)

        real_label[(real_label < self.opt.n_class_seen)] = 0
        real_label[(real_label >= self.opt.n_class_seen)] = 1

        sum_sample = len(real_label)
        correct_sample = torch.sum(pre_label == real_label)
        # acc = float(correct_sample) / float(sum_sample)
        # print('val binary acc: {:.4f}  correct_sample: {:.4f}  sum_sample: {:.4f}'.format(acc, correct_sample, sum_sample))
        return correct_sample, sum_sample

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        acc_top1 = 0
        if self.opt.acc_top1:
            # print("## todo ##")
            acc_per_class += float(torch.sum(test_label == predicted_label)) / float(test_label.size(0))
        else:
            for i in target_classes:
                idx = (test_label == i)
                acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
            acc_per_class /= target_classes.size(0)
        return acc_per_class 

    def val_gzsl(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.opt.batch_size):
            end = min(ntest, start+self.opt.batch_size)
            with torch.no_grad():
                if self.opt.cuda:
                    embed = self.netMap(test_X[start:end].cuda())
                    output = self.model(embed)
                else:
                    embed = self.netMap(test_X[start:end])
                    output = self.model(embed)
            _, predicted_label[start:end] = torch.max(output, 1)
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc
    


    def get_best_acc(self, acc_seen, acc_unseen, _best_seen, _best_unseen, _best_H):
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


class F_MLP(nn.Module):
    def __init__(self, opt):
        super(F_MLP, self).__init__()
        self.fc = nn.Linear(opt.embedSize, opt.n_class_all)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        embedding = self.fc(x)
        o = self.log_softmax(embedding)
        return o

class OOD_MLP(nn.Module):
    def __init__(self, opt):
        super(OOD_MLP, self).__init__()
        self.fc1 = nn.Linear(opt.x_dim, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.n_class_seen)

        self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        embedding= self.relu(self.fc1(x))
        embedding = self.fc2(embedding)
        o = self.logic(embedding)

        return o, embedding