import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_syn_feature_for_ood_train(netG, classes, attribute, num, opt):
    nclass = classes.size(0)
    # syn_feature = torch.FloatTensor(nclass * num, opt.x_dim)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(nclass * num, opt.attr_dim)
    syn_noise = torch.FloatTensor(nclass * num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.narrow(0, i * num, num).copy_(iclass_att.repeat(num, 1))
        syn_label.narrow(0, i * num, num).fill_(iclass)
    syn_noise.normal_(0, 1)
    output = netG(syn_noise, syn_att)
    return output, syn_label


def fig_ood_loss(map_label, logit_v, output, ood_criterion,opt):
    # 
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

    # energy
    # Ec_out = -torch.logsumexp(ood_logit, dim=1)
    # Ec_in = -torch.logsumexp(id_logit, dim=1)
    # ood_loss = opt.ood_loss_para*(torch.pow(F.relu(Ec_in-opt.m_in), 2).mean() + torch.pow(F.relu(opt.m_out-Ec_out), 2).mean())
    # return class_loss, ood_loss
    return class_loss

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
            energy_score[start:end] = output.max(dim=1).values
            # print(energy_score[start:end])
            start = end
        return energy_score

def compute_per_class_acc_ood(test_label, predicted_label, start, end):
    acc_per_class = 0
    for i in range(start, end):
        idx = (test_label == i)
        acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    acc_per_class /= (end - start)
    return acc_per_class 

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

def map_label_all(label, seenclasses, unseenclasses, _nclass_s):
    mapped_label = torch.LongTensor(label.size())
    nclass_s = _nclass_s
    for i in range(seenclasses.size(0)):
        mapped_label[label == seenclasses[i]] = i
    
    for j in range(unseenclasses.size(0)):
        mapped_label[label == unseenclasses[j]] = j + nclass_s
        
    return mapped_label

def print_net_para(_net):
    net = _net
    # print(net)


    # print(net.state_dict())

   
    # print(net.state_dict().keys())


    # # print(net.state_dict()["net1.bias"])
    # # print(net.state_dict()["net1.bias"].shape)

    for name, parameters in net.named_parameters():
        print(name,':', parameters, parameters.size())

def print_net_grad(_net):
    net = _net
    for name, parameters in net.named_parameters():
        print(name,':', parameters.grad, parameters.size())

def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    acc_per_class /= target_classes.size(0)
    return acc_per_class 

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
                    embed,_ = netMap(test_X[start:end].cuda())
                    output, _, _ = val_model(embed, seen_classes, unseen_classes)
                else:
                    embed,_ = netMap(test_X[start:end])
                    output, _, _ = val_model(embed, seen_classes, unseen_classes)
                # output = torch.nn.LogSoftmax(output)
            _, predicted_label[start:end] = torch.max(output, 1)
            start = end

        acc = compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

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

def calc_ood_loss(feature, label, map_label, opt, data_train, netG, netMap, netF_CLS, netOODMap, netOOD, net_sigmoid, final_criterion, KL_Loss, bi_criterion):

    syn_feature, syn_label = generate_test(netG, data_train.unseenclasses, data_train.attribute, opt.ood_train_syn_num, opt)
    syn_map_label = map_label_all(syn_label, data_train.seenclasses, data_train.unseenclasses, data_train.ntrain_class)
    netF_CLS.zero_grad()
    # netMap.zero_grad()
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


    embed, out_z = netMap(syn_feature)
    output, bi_feature, cls_logits = netF_CLS(embed, data_train.seenclasses, data_train.unseenclasses)
    final_cls_loss  = final_criterion(output, syn_label)


    embed_O = netOODMap(syn_feature)
    ood_output, ood_logits = netOOD(embed_O)
    energy_score, _ = torch.max(ood_output, dim=1, keepdim=True)

    # kl loss

    indices = torch.arange(start_select, end_select).cuda()
    label_teacher = torch.index_select(syn_label, dim=0, index=indices)
    label_student = torch.index_select(syn_label, dim=0, index=indices)


    # softmax kl loss
    student_logits = torch.index_select(cls_logits, dim=0, index=indices)
    teacher_logits = torch.index_select(ood_logits, dim=0, index=indices)
    # logits_distillation_loss = KL_Loss(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1))
    # logits_distillation_loss = batch_embed_distillation(teacher_logits, student_logits, label_teacher, label_student, bi_criterion)
    logits_distillation_loss = fig_single_distillation_loss(student_logits, teacher_logits)

    # # embed distillation loss
    embed_teacher = torch.index_select(embed_O, dim=0, index=indices)
    embed_student = torch.index_select(embed, dim=0, index=indices)
    # embed_distillation_loss = lrd_loss(embed_student, embed_teacher)

    embed_distillation_loss = batch_embed_distillation(embed_teacher, embed_student, label_teacher, label_student, bi_criterion)


    # kl_loss = KL_Loss(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1))
    # sigmoid
    Sequence_label = net_sigmoid(energy_score)
    OOD_confidence = 1 - Sequence_label
    Sequence_label = torch.cat((Sequence_label, OOD_confidence), 1)
    OOD_contrastive_loss = batch_cosine_similarity(F.softmax(bi_feature, dim=1), F.softmax(Sequence_label, dim=1), syn_label, bi_criterion)

    OOD_loss = OOD_contrastive_loss
    # OOD_loss = OOD_contrastive_loss + OOD_distillation_loss
    # distillation_loss = logits_distillation_loss
    # distillation_loss = embed_distillation_loss
    distillation_loss = embed_distillation_loss + logits_distillation_loss
    return final_cls_loss, OOD_loss, distillation_loss

def fig_single_distillation_loss(_bi_feature, _Sequence_label):
    # KL
    KL_Loss = nn.KLDivLoss(reduction="batchmean")      
    loss = KL_Loss(F.log_softmax(_bi_feature, dim=1), F.softmax(_Sequence_label, dim=1))


    # mse_loss = nn.MSELoss()
    # loss = mse_loss(_bi_feature, _Sequence_label)


    # l1_loss = nn.L1Loss()
    # loss = l1_loss(_bi_feature, _Sequence_label)


    # loss = js_divergence(_bi_feature, _Sequence_label)

    return loss

def kl_divergence(p, q):

    return (p * (p / q).log()).sum()

def js_divergence(p, q):

    p = F.softmax(p, dim=-1)  
    q = F.softmax(q, dim=-1)  
    m = (p + q) / 2
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2


def batch_embed_distillation(real_seen_feat, syn_seen_feat, real_seen_label, syn_seen_label, bi_criterion):
    # a_number = real_seen_feat.size(0)
    # b_number = syn_seen_feat.size(0)
    # a_embedding = real_seen_feat.unsqueeze(1).repeat(1, b_number, 1).view(-1, real_seen_feat.size(1))
    # b_embedding = syn_seen_feat.unsqueeze(0).repeat(a_number, 1, 1).view(-1, syn_seen_feat.size(1))

    # similarity = (torch.cosine_similarity(a_embedding, b_embedding) + 1) / 2
    # similarity = similarity.view(similarity.size(0), -1)

    # 
    real_seen_feat_norm = real_seen_feat / real_seen_feat.norm(dim=1).unsqueeze(1)
    syn_seen_feat_norm = syn_seen_feat / syn_seen_feat.norm(dim=1).unsqueeze(1)

    # 
    similarity = torch.matmul(real_seen_feat_norm, syn_seen_feat_norm.t())

    # [0, 1]
    similarity = (similarity + 1) / 2
    similarity = similarity.view(-1, 1)


    real_seen_label = real_seen_label.contiguous().view(1, -1)
    syn_seen_label = syn_seen_label.contiguous().view(-1, 1)
    
    # ground_truth_label
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