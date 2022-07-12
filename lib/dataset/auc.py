import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re
import random
import math
import cv2
import csv

#from lib.dataset.heatmap import *

class AUC(data.Dataset):
    def __init__(self, split = 'train', imSize=(224,224),aug=False):
        
        self.dataPath = './dataset/AUC/v1_cam1_no_split'
        self.imSize = imSize
        
        self.aug=aug
        self.seq_num=1
        self.raw_wh=[1920,1080]
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transformImg = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        self.transformFlow = transforms.Compose([
            transforms.Resize(self.imSize),
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.ToTensor(),
        ])
        
        
        if split == 'train':
            self.indices=self._get_db(True)
        else:
            self.indices=self._get_db(False)
        
        super(AUC).__init__()
        
        print('Loaded AUC dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            #im = Image.new("RGB", self.imSize, "white")

        return im


    def __getitem__(self, index):
        sample = self.indices[index]
        
        #load image
        img_seq=[]
        #sal_seq=[]
        for i in range(self.seq_num):
            img = self.loadImage(sample['image'][i])
            #img_sal=Image.open(sample['sal'][i]).convert('L')
            img=self.transformImg(img)
            img_seq.append(img)
        img_seq=torch.stack(img_seq,dim=0)
        img_seq=img_seq.permute(1,0,2,3)#(channel, t, h, w)
        
        img_seq=img_seq.squeeze()
        
        '''
        #load optical flow
        flow_seq=[]
        for i in range(self.seq_num-1):
            #flow=Image.open(sample['flow'][i])#.convert('L')
            flow=self.loadImage(sample['flow'][i])
            flow=self.transformFlow(flow)
            flow_seq.append(flow)
        flow_seq=torch.stack(flow_seq,dim=0)
        flow_seq=flow_seq.permute(1,0,2,3)#(channel, t, h, w)
        
        #load pose
        pose_xy=np.zeros((18*self.seq_num,2))
        
        for i in range(self.seq_num):
            subset=np.load(sample['subset'][i])
            candidate=np.load(sample['candidate'][i])
            for k in range(18):
                try:
                   if subset[0][k]==-1:
                      continue
                except IndexError as error:
                   continue
                   Logging.log_exception(error)
                else:
                   idx=int(subset[0][k])
                   pose_xy[i*18+k,:]=candidate[idx][0:2]
        
        pose_xy=pose_xy/self.raw_wh
        pose_xy=pose_xy.reshape((-1))
        pose_xy=torch.FloatTensor(pose_xy)
        '''
        
        img_label=sample['label']
        #img_label_cls=torch.LongTensor(img_label) #for multi class crossentropy
        img_label=np.minimum(img_label,1) # multiple class to 0-1
        #img_label=torch.LongTensor(img_label) #for multi class crossentropy
        img_label=torch.FloatTensor(img_label) #for binary class BCELoss

        return img_seq, img_label#,img_label_cls
    
        
    def __len__(self):
        return len(self.indices)
    
    def _get_db(self,is_train):
        if is_train:
            # use ground truth bbox
            gt_db = self._load_samples(True)
        else:
            # use bbox from detection
            gt_db = self._load_samples(False)
        return gt_db
    
    def _get_test_samples(self):
        
        csv_file='Test_data_list.csv'
        
        csvfile=open('%s/%s'%(self.dataPath,csv_file), newline='')
        reader = csv.DictReader(csvfile)
        
        test_labels={}
        for i in range(10):
           test_labels['%d'%i]=[]
        
        for row in reader:
            cls_name=row['Image'].split('/')[2]
            img_name=row['Image'].split('/')[3]
            img_idx=int(img_name.split('.')[0])
            int(row['Label'])
            for k in range(int(self.seq_num/2)):
                test_labels[row['Label']].append(img_idx-k)
            
        return test_labels
    
    def _load_samples(self,is_train=False):
        # get images and groundtruths(position of center of head, pitch,yaw,roll,rotation matrix)
        gt_db=[]

        if is_train:
           csv_file='Train_data_list.csv'
        else:
           csv_file='Test_data_list.csv'
        
        test_labels=self._get_test_samples()
        
        csvfile=open('%s/%s'%(self.dataPath,csv_file), newline='')
        reader = csv.DictReader(csvfile)
        
        
        for row in reader:
            
            #if len(gt_db)>1000:
            #   break
            image_path_list=[]
            flow_path_list=[]
            subset_path_list=[]
            cand_path_list=[]
            #hand_path_list=[]
            
            cls_name=row['Image'].split('/')[2]
            img_name=row['Image'].split('/')[3]
            img_idx=int(img_name.split('.')[0])
            
            if img_idx<self.seq_num:
               continue
           
            
            if is_train and int(row['Label'])>9:
               continue
            
            
            for k in range(self.seq_num):
                image_path_list.append(os.path.join(self.dataPath,'%s'%cls_name,'%d.jpg'%(img_idx-(self.seq_num-k)+1)))
                subset_path_list.append(os.path.join('%s_pose'%self.dataPath,'%s'%cls_name,'%d_subset.npy'%(img_idx-(self.seq_num-k)+1)))
                cand_path_list.append(os.path.join('%s_pose'%self.dataPath,'%s'%cls_name,'%d_candidate.npy'%(img_idx-(self.seq_num-k)+1)))
                
                if k==0:
                   continue
                flow_path_list.append(os.path.join('%s_flow'%self.dataPath,'%s'%cls_name,'%d.jpg'%(img_idx-(self.seq_num-k)+1)))
            
            for k in range(self.seq_num):
                if not os.path.exists(image_path_list[k]):
                   continue
                if not os.path.exists(subset_path_list[k]):
                   continue
                if not os.path.exists(cand_path_list[k]):
                   continue
            gt_db.append({
                'image':image_path_list,
                'flow':flow_path_list,
                'subset':subset_path_list,
                'candidate':cand_path_list,
                'label':np.array([int(row['Label'])]),
                })
                
        return gt_db
