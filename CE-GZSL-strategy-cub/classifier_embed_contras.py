import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler 
import sys
import torch.nn.functional as F
from ce_config import ce

class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _my_opt, _train_X, _train_Y, _train_Y_map, map_net, embed_size, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True):
        self.opt = _my_opt
        self.train_X =  _train_X 
        self.train_Y = _train_Y 
        self.train_Y_map = _train_Y_map
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label 
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.MapNet=map_net
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = embed_size
        self.cuda = _cuda
        self.model =  LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()
        
        self.input = torch.FloatTensor(_batch_size, _train_X.size(1))
        self.label = torch.LongTensor(_batch_size) 
        self.label_map = torch.LongTensor(_batch_size) 
      
        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        self.idx_seen = data_loader.seenclasses
        self.idx_unseen = data_loader.unseenclasses

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
            self.label_map = self.label_map.cuda()
            self.idx_seen = self.idx_seen.cuda()
            self.idx_unseen = self.idx_unseen.cuda() 


        # self.idx_seen, self.idx_unseen = self.label_split()
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        
        if generalized:
            self.acc_seen, self.acc_unseen, self.H, self.best_bi_ood_model_acc, self.best_bi_model_acc = self.fit()
        else:
            self.acc = self.fit_zsl()
    
    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label, _ = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                   
                embed, _=self.MapNet(self.input)
                output, _, _= self.model(embed, self.idx_seen, self.idx_unseen)
                loss = self.criterion(output, self.label)
                mean_loss += loss.data
                loss.backward()
                self.optimizer.step()
            acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if acc > best_acc:
                best_acc = acc
        print('Training classifier loss= %.4f' % (loss))
        return best_acc 

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        best_bi_ood_model_acc = 0
        best_bi_model_acc = 0
        for epoch in range(self.nepoch):
            loss_sum = 0.0
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label, _ = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                embed, _ = self.MapNet(self.input)
                output = self.model(embed)
                loss = self.criterion(output, self.label)
                loss_sum += loss

                loss.backward()
                self.optimizer.step()
            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            print('[%d/%d] Loss_final: %.4f ### seen acc: %.4f  unseen acc: %.4f' % (epoch + 1, self.nepoch, loss_sum, acc_seen, acc_unseen))
            if (acc_seen+acc_unseen)==0:
                print('a bug')
                H=0
            else:
                H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
        return best_seen, best_unseen, best_H, best_bi_ood_model_acc, best_bi_model_acc


    def fit_train_ood(self):
        self.ood_model.train()
        for p in self.ood_model.parameters():  # reset requires_grad
            p.requires_grad = True
        for epoch in range(ce.epoch_ood):
            for i in range(0, self.ntrain, self.batch_size):      
                self.ood_model.zero_grad()
                batch_input, batch_label, batch_label_map = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                self.label_map.copy_(batch_label_map)

                embed, _ = self.MapNet(self.input)
                output, logit_v = self.ood_model(embed)

             
                id_output = torch.FloatTensor().cuda() 
                id_label = torch.LongTensor().cuda() 
                id_logit = torch.FloatTensor().cuda() 

                ood_logit = torch.FloatTensor().cuda() 

                # for j in self.seenclasses:    
                idj = (self.label_map < 150)   
                id_label = torch.cat((id_label, self.label_map[idj]), 0)
                id_logit = torch.cat((id_logit, logit_v[idj]), 0)
                id_output = torch.cat((id_output, output[idj]), 0)

                # ID 
                # for k in self.unseenclasses:
                idx = (self.label_map >= 150)
                ood_logit = torch.cat((ood_logit, logit_v[idx]), 0)
                                   
                class_loss = self.criterion(id_output, id_label)

                # energy
                Ec_out = -torch.logsumexp(ood_logit, dim=1)
                Ec_in = -torch.logsumexp(id_logit, dim=1)
                # print(Ec_out)
                # print(Ec_in)
                ood_loss = ce.ood_loss_para*(torch.pow(F.relu(Ec_in-ce.m_in), 2).mean() + torch.pow(F.relu(ce.m_out-Ec_out), 2).mean())

                # print('class_loss : {:.4f}  ood_loss : {:.4f}'.format(class_loss.data, ood_loss.data))
                loss = class_loss + ood_loss

                loss.backward()
                self.ood_optimizer.step()



    def fit_classifier(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        best_bi_ood_model_acc = 0
        best_bi_model_acc = 0

        self.ood_model.eval()
        for p in self.ood_model.parameters():  # reset requires_grad
            p.requires_grad = False
        for epoch in range(self.nepoch):
            bi_model_sum_correct_sample = 0
            bi_model_sum_sample = 0
            bi_ood_model_sum_correct_sample = 0
            bi_ood_model_sum_sample = 0
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label, batch_label_map= self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                self.label_map.copy_(batch_label_map)

                embed, _ = self.MapNet(self.input)
                output, _embed, bi_feature = self.model(embed, self.idx_seen, self.idx_unseen)
                loss = self.criterion(output, self.label)
############################################################################################################################################
                # 
                _, logit_v = self.ood_model(embed)
                energy_score = -torch.logsumexp(logit_v, dim=1)
                threshold = ce.threshold
                di_label = torch.LongTensor(energy_score.size()).cuda()
                in_idx = (energy_score < threshold)
                out_idx = (energy_score >= threshold)
                di_label[in_idx] = 0
                di_label[out_idx] = 1
              
                # bi_feature = bi_feature.squeeze(dim=-1)
                # bi_loss = ce.bi_loss_para * self.bi_BCEWithLogitsLoss(bi_feature,di_label)
                bi_loss = ce.bi_loss_para * self.bi_CrossEntropyLoss(bi_feature,di_label)
                # print('loss : {:.4f}  bi_loss : {:.4f}'.format(loss.data, bi_loss.data))
                loss += bi_loss
############################################################################################################################################
                loss.backward()
                self.optimizer.step()

############################################################################################################################################
                # ood_model
                bi_ood_model_correct_sample, bi_ood_model_samples = self.bi_acc(di_label, self.label_map)
                bi_ood_model_sum_correct_sample += bi_ood_model_correct_sample
                bi_ood_model_sum_sample += bi_ood_model_samples
                # model
                _, pre_di_label = torch.max(bi_feature, 1)
                bi_model_correct_sample, bi_model_samples = self.bi_acc(pre_di_label, self.label_map)
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

            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)         
            if (acc_seen+acc_unseen)==0:
                print('a bug')
                H=0
            else:
                H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
        return best_seen, best_unseen, best_H, best_bi_ood_model_acc, best_bi_model_acc
        # return best_seen, best_unseen, best_H, best_bi_ood_model_acc
        # return best_seen, best_unseen, best_H

    def bi_acc(self, pre_label, map_label):
        real_label = torch.LongTensor(len(map_label)).cuda()
        real_label.copy_(map_label)

        real_label[(real_label < 150)] = 0
        real_label[(real_label >= 150)] = 1

        sum_sample = len(real_label)
        correct_sample = torch.sum(pre_label == real_label)
        # acc = float(correct_sample) / float(sum_sample)
        # print('val binary acc: {:.4f}  correct_sample: {:.4f}  sum_sample: {:.4f}'.format(acc, correct_sample, sum_sample))
        return correct_sample, sum_sample


    # def label_split(self):
    #     idx_seen = torch.LongTensor().cuda()
    #     idx_unseen = torch.LongTensor().cuda()
    #     for i in self.seenclasses:
    #         ids = (self.label == i)
    #         ids = torch.nonzero(ids == True)
    #         idx_seen = torch.cat((idx_seen, ids), 0)
    #     for j in self.unseenclasses:
    #         idu = (self.label == j)
    #         idu = torch.nonzero(idu == True)
    #         idx_unseen = torch.cat((idx_unseen, ), 0)

    #     return idx_seen.squeeze(), idx_unseen.squeeze()

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            self.train_Y_map = self.train_Y_map[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
                Y_rest_part_map = self.train_Y_map[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            self.train_Y_map = self.train_Y_map[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            Y_new_part_map = self.train_Y_map[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0), torch.cat((Y_rest_part_map, Y_new_part_map), 0)
            else:
                return X_new_part, Y_new_part, Y_new_part_map
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end], self.train_Y_map[start:end]


    def val_gzsl(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    embed, _ = self.MapNet(test_X[start:end].cuda())
                    output = self.model(embed)
                else:
                    embed, _ = self.MapNet(test_X[start:end])
                    output = self.model(embed)
            _, predicted_label[start:end] = torch.max(output, 1)
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
        acc_per_class /= target_classes.size(0)
        return acc_per_class 

    # test_label is integer 
    def val(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    embed, _ = self.MapNet(test_X[start:end].cuda())
                    output, _, _ = self.model(embed, self.idx_seen, self.idx_unseen)
                else:
                    embed, _ = self.MapNet(test_X[start:end])
                    output, _, _ = self.model(embed, self.idx_seen, self.idx_unseen)
            _, predicted_label[start:end] = torch.max(output, 1)
            start = end

        acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = float(torch.sum(test_label[idx]==predicted_label[idx])) / float(torch.sum(idx))
        return acc_per_class.mean() 

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o  

        



class LINEAR_OOD(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_OOD, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        lo = self.fc(x)
        o = self.logic(lo)
        return o,lo 
