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
    def __init__(self, Backbones, FCs, n_way, n_support):
        super(eProtoNetFC, self).__init__(n_way, n_support)
        self.loss_fn  = nn.CrossEntropyLoss()
        self.Backbones = Backbones
        self.FCs = FCs

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

    def set_forward_loss(self, x1, x2):
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
            
            loss = loss_1 + loss_2 + loss_3 + loss_4

        return loss
    
    def feature_extractor(self, x, resnet_idx):
        x = Variable(x.to(self.device))
        x = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
        z_all = self.Backbones[resnet_idx].forward(x) ## Resnet to list, feature : 0,1,2
        return z_all

    def fc_layer(self, z_all, fc_idx):
        z_all       = self.FCs[fc_idx].forward(z_all.to(self.device))
        z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        return z_all

    def parse_feature(self, z_all):
        z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query
    

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
                    print('Epoch {:d} | Training Phase: AGG | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))

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
                    print('Epoch {:d} | Training Phase: EPI | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
                
#         ###

#         for i, (x1, _ ,x2, _ ) in enumerate(train_loader[0], train_loader[1]):
#             self.n_query = x.size(1) - self.n_support           
#             if self.change_way:
#                 self.n_way  = x.size(0)
#             optimizer.zero_grad()
#             loss = self.set_forward_loss( x )
#             loss.backward()
#             optimizer.step()
#             avg_loss = avg_loss+loss.item()

#             if i % print_freq==0:
#                 #print(optimizer.state_dict()['param_groups'][0]['lr'])
#                 print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))


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
