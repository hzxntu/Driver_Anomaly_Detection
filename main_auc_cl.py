import math, shutil, os, time, argparse
import numpy as np
import scipy.io as sio

os.environ['CUDA_VISIBLE_DEVICES']='0,3'

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from lib.dataset.auc_seq import AUC
from lib.dataset.auc_seq_cl import AUC_CL
from lib.model.res_bl_cl import RES_BL_CL
from lib.model.res_flow import RES_FLOW
from lib.model.res_pose import RES_POSE
from lib.core.loss import CLSLoss,ContrastiveLoss
from lib.utils.utils import knn_score

from metrics import cal_auc,cal_pfr,cal_cm,cal_auc_sklearn,draw_roc


from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
parser.add_argument('--sink', type=str2bool, nargs='?', const=True, default=False, help="Just sink and terminate.")
parser.add_argument('--reset', type=str2bool, nargs='?', const=True, default=False, help="Start from scratch (do not load).")
args = parser.parse_args()

# Change there flags to control what happens.
doLoad = not args.reset # Load checkpoint at the beginning
doTest = args.sink # Only run test, no training
doPrediction= True # Only prediction, no testing no training #hzx added

workers = 16
epochs = 20
#batch_size = torch.cuda.device_count()*32 # Change if out of cuda memory
batch_size=64

cls_num=2

base_lr = 0.0005
momentum = 0.9
weight_decay = 0.00005 #1e-4
print_freq = 10
prec1 = 0
best_prec1 = 0.5
lr = base_lr

count_test = 0
count = 0



def main():
    global args, best_prec1, weight_decay, momentum
    

    
    cudnn.benchmark = True 
    
    model = RES_BL_CL()
    
    criterion = ContrastiveLoss().cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    model = torch.nn.DataParallel(model,device_ids=[0,1]).cuda()  

    epoch = 0
    if doLoad:
        saved = load_checkpoint()
        if saved:
            print('Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...' % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                model.module.load_state_dict(state,strict=True)
            except:
                model.load_state_dict(state,strict=True)
            
            epoch = saved['epoch']
            #best_prec1 = saved['best_prec1']
            #print(best_prec1)
        else:
            print('Warning: Could not read checkpoint!')
    
    
    dataTrain = AUC(split='train')
    dataVal = AUC(split='test')
    dataTrain_cl = AUC_CL(split='train')
   
    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=batch_size, shuffle=True,
        num_workers=workers)
        
    train_loader_cl = torch.utils.data.DataLoader(
        dataTrain_cl,
        batch_size=batch_size, shuffle=True,
        num_workers=workers)
    
    #preprocessing 
    auc, feature_space = get_score(model, train_loader, val_loader)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    
    # Quick test
    if doTest:
        return
   
    center = torch.FloatTensor(feature_space).mean(dim=0)
    center = F.normalize(center, dim=-1)
    center = center.cuda()
    
    for epoch in range(0, epoch):
        adjust_learning_rate(optimizer, epoch)
        
    for epoch in range(epoch, epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        running_loss = train(train_loader_cl, model,criterion, optimizer, center,epoch)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        
        # evaluate on validation set
        auc, feature_space = get_score(model, train_loader, val_loader)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
        
        center = torch.FloatTensor(feature_space).mean(dim=0)
        center = F.normalize(center, dim=-1)
        center = center.cuda()
        
        prec1=auc        

        # remember best prec and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch ,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def get_score(model, train_loader, test_loader):
    train_feature_space = []
    model.eval()
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.cuda()
            features,_ = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    test_labels = []
    #test_labels_cls=[]
    with torch.no_grad():
        for (imgs, labels) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.cuda()
            features,_ = model(imgs)
            test_feature_space.append(features)
            test_labels.append(labels)
            #test_labels_cls.append(labels_cls)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()
        #test_labels_cls = torch.cat(test_labels_cls, dim=0).cpu().numpy()
    
    center = np.mean(train_feature_space,axis=0)
    center = center / np.linalg.norm(center)
    train_feature_space_c=train_feature_space-center
    test_feature_space_c=test_feature_space-center 
    
    #distances = knn_score(train_feature_space_c, test_feature_space_c)
    distances = knn_score(train_feature_space, test_feature_space)
    
    '''
    center = np.mean(train_feature_space,axis=0)
    center = center / np.linalg.norm(center)
    #train_feature_space_c=train_feature_space-center
    #test_feature_space_c=test_feature_space-center 

    similarities=np.matmul(test_feature_space,center.reshape(test_feature_space.shape[1],1))
    #max_1=np.max(similarities,axis=1)
    distances=similarities
    '''
    
    
    auc = roc_auc_score(test_labels, distances)
    #draw_roc(test_labels,distances)
    
    '''
    save_path='./predict/auc_bl_cl_ncent_trainall_multicls'
    if not os.path.exists(save_path):
       os.mkdir(save_path)
    #np.save('%s/test_labels.npy'%save_path,test_labels)
    np.save('%s/distances.npy'%save_path,distances)
    #np.save('%s/features.npy'%save_path,test_feature_space)
    #np.save('%s/test_labels_cls.npy'%save_path,test_labels_cls)
    '''
    
    
    '''
    for i in range(2,16):
        pred=np.zeros_like(distances)
        pred[np.where(distances>(i/100))]=1
        cal_cm(test_labels,pred,i)
    '''

    return auc, train_feature_space

def train(train_loader, model, criterion,optimizer, center, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    total_loss, total_num = 0.0, 0

    # switch to train mode
    model.train()

    end = time.time()
    #loss_file = open("./loss/nall_aff/loss_%03d.txt"%epoch,"w")
    i=0

    #for i, (input_img,input_flow,input_pose,target) in enumerate(train_loader):
    for (img1, img2, labels) in tqdm(train_loader, desc='Train...'):
        
        # measure data loading time
        data_time.update(time.time() - end)

        img1, img2 = img1.cuda(), img2.cuda()
        labels=labels.cuda()
        optimizer.zero_grad()

        out_1,out_p_1 = model(img1)
        out_2,out_p_2 = model(img2)

        loss = criterion(out_1,out_2,center,out_p_1,out_p_2,labels) 

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)

        # measure accuracy and record loss
        losses.update(loss.item(), img1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        '''
        if i%100==0:
            print('Epoch (train): [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses))
        i+=1
        '''

        #loss_file.writelines('%03f/%03f '%(losses.val,losses.avg))
    #loss_file.close()
    return total_loss / (total_num)

def validate(val_loader, model, criterion, epoch):
    #global count_test
    batch_time = AverageMeter()
    losses = AverageMeter()
    auc=AverageMeter()
    
    auc_all=[]
    
    target_all=[]
    output_all=[]
    target_v_all=[]

    end = time.time()
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
         for i, (input_img, input_flow,input_pose,target) in enumerate(val_loader):
    
            input_img=input_img.cuda()
            input_flow=input_flow.cuda()
            input_pose=input_pose.cuda()
            target = target.cuda()
            
            input_img = torch.autograd.Variable(input_img, requires_grad = False)
            input_flow = torch.autograd.Variable(input_flow, requires_grad = False)
            input_pose = torch.autograd.Variable(input_pose, requires_grad = False)
            target = torch.autograd.Variable(target, requires_grad = False)
            
            # compute output
            outputs= model(input_img)

            loss = criterion(outputs, target)

            num_images = input_img.size(0)
            losses.update(loss.item(), num_images)
            
            #target_cpu=target.cpu()
            #target_onehot=np.zeros((target_cpu.shape[0],cls_num))
            #target_onehot[np.arange(target_cpu.shape[0]),target_cpu] = 1
            
            #target_v_all.append(target_onehot)
            output_all.append(outputs.cpu().numpy())
            target_all.append(target.cpu().numpy())
            
            #roc_auc=cal_auc(target_onehot,outputs.cpu().numpy(),cls_num)
            
            #auc.update(roc_auc['macro'],num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if i % 100 == 0:
                print('Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                          i, len(val_loader), batch_time=batch_time))
          
         #target_v_all=np.concatenate(target_v_all,axis=0)
         output_all=np.concatenate(output_all,axis=0)
         target_all=np.concatenate(target_all,axis=0)
         
         #roc_auc=cal_auc(target_v_all,output_all,cls_num)
         auc_score=cal_auc_sklearn(target_all.squeeze(),output_all.squeeze())
         
         #for avg in ['weighted','macro','micro']:
         #    p_score,f_score,r_score=cal_pfr(target_all.squeeze(),output_all,average=avg)
         #    print('%s:  '%avg, 'Precision: ',p_score, 'F1: ', f_score, 'Recall: ',r_score)
         #cal_cm(target_all.squeeze(),output_all)
    
    perf_indicator=auc_score#roc_auc['macro']
    
    #print('AUC:',roc_auc['macro'],roc_auc['micro'])
    print('AUC: ',auc_score)
    
    return perf_indicator

CHECKPOINTS_PATH = './output/auc_bl_cl_trainall'

def load_checkpoint(filename='best_checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_' + filename)
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


if __name__ == "__main__":
    
    main()
    print('DONE')
