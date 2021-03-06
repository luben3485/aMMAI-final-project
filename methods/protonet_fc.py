# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

import utils

class ProtoNetFC(MetaTemplate):
    def __init__(self, model_func, fc, n_way, n_support):
        super(ProtoNetFC, self).__init__( model_func,  n_way, n_support)
        self.loss_fn  = nn.CrossEntropyLoss()
        self.fc = fc

    def set_forward(self,x,is_feature = False):
        ### jiafong
        #z_support, z_query  = self.parse_feature(x,is_feature)
        
        z_all = self.feature_extractor(x)
        z_all = self.fc_layer(z_all)
        z_support, z_query  = self.parse_feature(z_all) 

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )
        
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())
        scores = self.set_forward(x)
        loss = self.loss_fn(scores, y_query)

        return loss
    
    def feature_extractor(self,x):
        x = Variable(x.to(self.device))
        x = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
        z_all = self.feature.forward(x)
        return z_all

    def fc_layer(self, z_all):
        z_all       = self.fc.forward(z_all.to(self.device))
        z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        return z_all

    def parse_feature(self, z_all):
        z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query
    



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
