import argparse
import os
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable
import torch.nn.functional as F

import cv2

import sys
sys.path.append('/home/automan/huzhongxu/codes/pytorchvideo')

from pytorchvideo.models.hub import slow_r50,i3d_r50
from pytorchvideo.models.head import create_res_basic_head, create_res_roi_pooling_head

sys.path.append('/home/automan/huzhongxu/codes/3D-ResNets-PyTorch')
from models import resnet, resnet2p1d, pre_act_resnet, wide_resnet, resnext, densenet

from lib.model.Resnets import *
from lib.model import resnet_dad


def _resnet(pretrained=True,model_depth=18,n_classes=700,n_input_channels=3,resnet_shortcut='B',conv1_t_size=7,conv1_t_stride=1,no_max_pool=False,resnet_widen_factor=1.0):
   model_name='resnet'
   model=resnet.generate_model(model_depth=model_depth,
                                      n_classes=n_classes,
                                      n_input_channels=n_input_channels,
                                      shortcut_type=resnet_shortcut,
                                      conv1_t_size=conv1_t_size,
                                      conv1_t_stride=conv1_t_stride,
                                      no_max_pool=no_max_pool,
                                      widen_factor=resnet_widen_factor)
   if pretrained:
      pretrain_path='../3D-ResNets-PyTorch/pretrained/data/r3d%d_K_200ep.pth'%(model_depth)
      print('loading pretrained model {}'.format(pretrain_path))
      pretrain = torch.load(pretrain_path, map_location='cpu')

      model.load_state_dict(pretrain['state_dict'])
      
      '''
      tmp_model = model
      if model_name == 'densenet':
            tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
                                             n_finetune_classes)
      else:
            tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                     n_finetune_classes)
      '''
   return model

def _resnet_dad(pretrained=True,model_depth=18,input_size=112,clip_num=16,resnet_shortcut='A'):
    model = resnet_dad.resnet18(
	            output_dim=128,
	            sample_size=input_size,
	            sample_duration=clip_num,
	            shortcut_type=resnet_shortcut,
	            tracking=True,
	            pre_train=False
	        )
	        
    if pretrained:
       pretrain_path='./pretrained/DAD/DAD_pretrained_files/DAD_pretrained_models/best_model_resnet_front_depth.pth'
       print('loading pretrained model {}'.format(pretrain_path))
       pretrained_dict = torch.load(pretrain_path)['state_dict']
       pretrained_dict = {k.replace('module.',''): v for k, v in pretrained_dict.items() if k in pretrained_dict}
       #try:
       #model.module.load_state_dict(pretrain['state_dict'],strict=True)
       #except:
       model.load_state_dict(pretrained_dict,strict=True)
    
    return model

class RES_BL(nn.Module):
    def __init__(self):
        super(RES_BL, self).__init__()
        
        #self.backbone=slow_r50(pretrained=True)
        #self.backbone=i3d_r50(pretrained=True)
        #self.backbone=_resnet(pretrained=True,model_depth=18)
        #self.backbone=resnet18(pretrained=True)
        self.backbone=_resnet_dad(pretrained=True)
        
        '''
        self.head = create_res_basic_head(
            in_features=2048,
            out_features=10,
            pool=nn.AvgPool3d,
            output_size=(1, 1, 1),
            pool_kernel_size=(4, 7, 7),
            dropout_rate=0.0,
            activation=None,
            output_with_global_average=True,
        )
        '''
        self.fc=nn.Linear(512, 1)
        

        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        
        #encoding
        #res_list=self.backbone(x)

        #x_norm= F.normalize(res_list, dim=-1)
        #x=self.fc(res_list)
        
        x,x_norm=self.backbone(x)
        x=self.fc(x)
        
        return x,x_norm#,node#,g_map
