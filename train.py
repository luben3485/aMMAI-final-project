import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import glob


import configs
import backbone

from methods.baselinetrain import BaselineTrain
from methods.protonet import ProtoNet
from methods.protonet_fc import ProtoNetFC
from methods.e_protonet_fc import eProtoNetFC
from methods.relationnet import RelationNet
from methods.gnnnet import GnnNet

from io_utils import model_dict, parse_args, get_resume_file  
from datasets import miniImageNet_few_shot, cifar100_few_shot

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')     

    max_acc = 0
    for epoch in range(start_epoch,stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader,  optimizer ) 

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop( val_loader)
        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        
    return model

def episodic_training(base_loaders, val_loaders, model, optimization, start_epoch, stop_epoch, params, train_phase, agg_i=None):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')     

    max_acc = 0
    for epoch in range(start_epoch,stop_epoch):
        model.train()
        if train_phase == 'epi':
            model.train_loop_epi(epoch, base_loaders,  optimizer ,train_phase=train_phase, agg_i=None) 
        elif train_phase == 'agg' and agg_i != None: 
            model.train_loop_epi(epoch, base_loaders,  optimizer ,train_phase=train_phase, agg_i=agg_i) 

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc_1 = model.test_loop_epi( val_loaders[0])
        acc_2 = model.test_loop_epi( val_loaders[1])
        acc = (acc_1 + acc_2) / 2
        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        
    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')
    task   = params.task

    image_size = 224
    optimization = 'Adam'

    base_loaders = []
    val_loaders = []

    if params.method in ['baseline'] :
        if task in ["fsl", "cdfsl-single"]:
            datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = 16)
            base_loaders.append(datamgr.get_data_loader(aug = params.train_aug ))
            val_loaders.append(None)
        else:
            miniImageNet_datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = 16)
            cifar100_datamgr = cifar100_few_shot.SimpleDataManager(image_size, batch_size = 16)

            base_loaders.append(miniImageNet_datamgr.get_data_loader(aug = params.train_aug ))
            val_loaders.append(None)
            base_loaders.append(cifar100_datamgr.get_data_loader( aug = params.train_aug ))
            val_loaders.append(None)

        model           = BaselineTrain( model_dict[params.model], params.num_classes)

    elif params.method in ['protonet', 'protonet_fc', 'e_protonet_fc', 'relationnet', 'gnnnet']:
        n_query = max(1, int(8* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 

        if task in ["fsl", "cdfsl-single"]:
            datamgr     = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="train",  **train_few_shot_params)
            val_datamgr = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="val",  **test_few_shot_params)

            base_loaders.append(datamgr.get_data_loader(aug = params.train_aug ))
            val_loaders.append(val_datamgr.get_data_loader(aug = False))

        else:
            mini_ImageNet_datamgr     = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="train",  **train_few_shot_params)
            mini_ImageNet_val_datamgr = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="val",  **test_few_shot_params)
            cifar100_datamgr          = cifar100_few_shot.SetDataManager(image_size, n_query = n_query, mode="train",  **train_few_shot_params)
            cifar100_val_datamgr      = cifar100_few_shot.SetDataManager(image_size, n_query = n_query, mode="val",  **test_few_shot_params)


            base_loaders.append(mini_ImageNet_datamgr.get_data_loader(aug = params.train_aug ))
            val_loaders.append(mini_ImageNet_val_datamgr.get_data_loader(aug = False))
            base_loaders.append(cifar100_datamgr.get_data_loader(aug = params.train_aug ))
            val_loaders.append(cifar100_val_datamgr.get_data_loader(aug = False))

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'protonet_fc':
            model           = ProtoNetFC( model_dict[params.model], backbone.FC(512,128), **train_few_shot_params )
        elif params.method == 'e_protonet_fc':
            #ResNets = [ model_dict[params.model], model_dict[params.model], model_dict[params.model]]
            #FCs = [backbone.FC(512,256), backbone.FC(512,256), backbone.FC(512,256)]
            #model           = eProtoNetFC( ResNets, FCs, backbone.FC(512,256), **train_few_shot_params )
            model           = eProtoNetFC(model_dict[params.model](), model_dict[params.model](), model_dict[params.model](),backbone.FC(512,128),backbone.FC(512,128),backbone.FC(512,128), **train_few_shot_params )
        
        elif params.method == 'relationnet':
            model           = RelationNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'gnnnet':
            model           = GnnNet( model_dict[params.model], **train_few_shot_params )
    
    else:
       raise ValueError('Unknown method')

    model = model.cuda()
    save_dir =  configs.save_dir

    task_path = 'single' if task in ["fsl", "cdfsl-single"] else 'multi'
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(save_dir, task_path, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if task in ["fsl", "cdfsl-single"]:
        model = train(base_loaders[0], val_loaders[0], model, optimization, start_epoch, stop_epoch, params)
    else:
        if params.method == 'e_protonet_fc':


            for i in range(2):
                print('now source domain ', i, ":")
                model = episodic_training(base_loaders, val_loaders, model, optimization, start_epoch, stop_epoch // 2, params, train_phase='agg', agg_i = i)

            model = episodic_training(base_loaders, val_loaders, model, optimization, start_epoch, stop_epoch, params, train_phase='epi')

        else:
            stop_epoch = stop_epoch // 2
            for i in range(2):
                print('now source domain ', i, ":")
                model = train(base_loaders[i], val_loaders[i], model, optimization, start_epoch, stop_epoch, params)
