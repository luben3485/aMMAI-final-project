# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.e_meta_template import MetaTemplate

import utils

class eProtoNetFC(MetaTemplate):
    def __init__(self, b0,b1,b2,fc0,fc1,fc2, n_way, n_support):
        super(eProtoNetFC, self).__init__(n_way, n_support)
        self.loss_fn  = nn.CrossEntropyLoss()
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.fc0 = fc0
        self.fc1 = fc1
        self.fc2 = fc2

    def set_forward(self, x, resnet_idx, fc_idx):
        ### jiafong
        #z_support, z_query  = self.parse_feature(x,is_feature)
        
        z_all = self.feature_extractor(x, resnet_idx)
        z_all = self.fc_layer(z_all, fc_idx)
        z_support, z_query = self.parse_feature(z_all)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        return scores

    def set_forward_loss(self, x1, x2=None):
        if x2 == None:
            y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
            y_query = Variable(y_query.cuda())
            scores = self.set_forward(x1, resnet_idx=0, fc_idx=0)
            loss = self.loss_fn(scores, y_query)
        
        elif x2 != None:
            y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
            y_query = Variable(y_query.cuda())
            
            scores_epic1 = self.set_forward(x1, resnet_idx=1, fc_idx=0)
            scores_epic2 = self.set_forward(x2, resnet_idx=2, fc_idx=0)
            scores_epif1 = self.set_forward(x1, resnet_idx=0, fc_idx=1)
            scores_epif2 = self.set_forward(x2, resnet_idx=0, fc_idx=2)

            loss_1 = self.loss_fn(scores_epic1, y_query)
            loss_2 = self.loss_fn(scores_epic2, y_query)
            loss_3 = self.loss_fn(scores_epif1, y_query)
            loss_4 = self.loss_fn(scores_epif2, y_query)
            
            loss = (loss_1 + loss_2 + loss_3 + loss_4)/4

        return loss
    
    def feature_extractor(self, x, resnet_idx):
        x = Variable(x.to(self.device))
        x = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
        #z_all = self.Backbones[resnet_idx].forward(x) ## Resnet to list, feature : 0,1,2
        if resnet_idx == 0:
            z_all = self.b0.forward(x)
        elif resnet_idx == 1:
            z_all = self.b1.forward(x)
        elif resnet_idx == 2:    
            z_all = self.b2.forward(x)
        else:
            assert False
        return z_all

    def fc_layer(self, z_all, fc_idx):
        #z_all       = self.FCs[fc_idx].forward(z_all.to(self.device))
        if fc_idx == 0:
            z_all = self.fc0.forward(z_all.to(self.device))
        elif fc_idx == 1:
            z_all = self.fc1.forward(z_all.to(self.device))
        elif fc_idx == 2: 
            z_all = self.fc2.forward(z_all.to(self.device))
        else:
            assert False
        z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        return z_all

    def parse_feature(self, z_all):
        z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query
    
    def correct(self, x):       
        scores  = self.set_forward(x,0,0)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)
    
    def test_loop_epi(self, test_loader, record = None):
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this*100  )

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        return acc_mean

    def train_loop_epi(self, epoch, train_loader, optimizer , train_phase, agg_i=None):
        """
        
        agg_i should be the domain idx in aggregation training.
        
        """
        print_freq = 10

        avg_loss=0
        
        if train_phase == 'agg' or train_phase == None:
            assert agg_i != None
        
            for i, (x, _) in enumerate(train_loader[agg_i]):
                self.n_query = x.size(1) - self.n_support           
                if self.change_way:
                    self.n_way  = x.size(0)
                optimizer.zero_grad()
                loss = self.set_forward_loss(x, None)
                loss.backward()
                optimizer.step()
                avg_loss = avg_loss+loss.item()

                if i % print_freq==0:
                    print('Epoch {:d} | Training Phase: AGG | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader[agg_i]), avg_loss/float(i+1)))

        elif train_phase == 'epi':
            
            for i, ((x1, _ ),( x2, _)) in enumerate(zip(train_loader[0], train_loader[1])):
                
                assert x1.size(1) == x2.size(1)
                self.n_query = x1.size(1) - self.n_support
                if self.change_way:
                    assert x1.size(0) == x2.size(0)
                    self.n_way  = x1.size(0)
                optimizer.zero_grad()
                loss = self.set_forward_loss(x1, x2)
                loss.backward()
                optimizer.step()
                avg_loss = avg_loss+loss.item()
                if i % print_freq==0:
                    print('Epoch {:d} | Training Phase: EPI | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader[0]), avg_loss/float(i+1)))


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
