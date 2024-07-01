from asyncore import write
import os
from sre_parse import SPECIAL_CHARS
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb
import time

from test_util import test_calculate_metric, test_calculate_metric_LA
from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, update_ema_variables

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/userdisk1/qjzeng/semi_seg/preprocess/', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='exp', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int,  default=5000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int,  default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=16, help='trained samples')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--lambda_', type=float, default=0.2, help='balance loss')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args()



def random_mask(img, mask_ratio):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    img_mask = torch.ones(batch_size, channel, img_x, img_y, img_z).cuda()
    label_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    for i in range(batch_size):
        w = np.random.randint(0, 96 - patch_pixel_x)
        h = np.random.randint(0, 96 - patch_pixel_y)
        z = np.random.randint(0, 96 - patch_pixel_z)
        label_mask[i, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
        img_mask[i, :, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    return img_mask.long(), label_mask.long()



def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

'''
def get_cut_mask(out, thres_h=0.9, thres_l=0.5, nms=0):

    probs = F.softmax(out, 1)
    
    masks_h = (probs >= thres_h).type(torch.int64)
    masks_h = masks_h[:, 1, :, :, :].contiguous()
    
    masks_l = (probs >= thres_l).type(torch.int64)
    masks_l = masks_l[:, 1, :, :, :].contiguous()
    
    if nms == 1:
        masks_h = LargestCC_pancreas(masks_h)
        masks_l = LargestCC_pancreas(masks_l)
        
    return masks_h.long(), masks_l.long()
'''

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda()

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2
maxdice1 = 0.

testset = LAHeart(train_data_path, split='test')
test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    model = nn.DataParallel(model)
    model = model.cuda()
    
    
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    maxdice1 = 0.
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)


    #val_dice, maxdice1, max_flag = test(model, model, test_loader, maxdice1)
    
    
    for epoch_num in iterator:
        for _, (sampled_batch, mim_mask) in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            mim_mask = mim_mask[:args.labeled_bs].unsqueeze(1).cuda()
            
            #----------------------learn a main decoder-------------------------#

            # cutmix for main decoder
            img_a, img_b = volume_batch, torch.flip(volume_batch, dims=[0])
            lab_a, lab_b = label_batch,  torch.flip(label_batch, dims=[0])
            
            with torch.no_grad():
                img_mask, _ = context_mask(img_a, args.mask_ratio)

            cutmix_batch = img_a * img_mask + img_b * (1 - img_mask)
            cutmix_label = lab_a * img_mask + lab_b * (1 - img_mask)
            num_cutmix   = cutmix_batch.shape[0]
            
            # forward of main decoder
            main_outputs = model(torch.cat((cutmix_batch, volume_batch), dim=0))
            
            # pseudo-label of main decoder
            main_outputs_ = main_outputs.detach()
            main_prob     = F.softmax(main_outputs_, dim=1)
            
            if iter_num > 1000:
                threshold = 0.5
            else:
                threshold = 0.2
            
            main_ps_lab   = main_prob[num_cutmix:,1,:,:,:] > threshold
            main_ps_lab   = main_ps_lab.unsqueeze(1).detach().int()
            
            
            #----------------------learn a mim decoder-------------------------#
            #if iter_num > 1000:
            #    mask_region = main_ps_lab
            #else:
            #    mask_region = mim_mask
            # masked input for mim decoder
            mask_region = main_ps_lab
            mim_batch = volume_batch * (1 - mask_region)
            
            # forward of mim decoder
            mim_outputs = model.module.mim_forward(mim_batch)           
            
            
            
            #----------------------learn an aux decoder-------------------------#
            
            # reconstructed input for aux decoder
            mim_outputs_ = mim_outputs.detach()
            re_batch = volume_batch * (1 - mask_region) + mim_outputs_.detach() * mask_region
            
            # forward of aux decoder
            aux_outputs = model.module.aux_forward(re_batch)             
            
            
            
            #----------------------loss calculation-------------------------#
            # loss for the main decoder
            loss_main_ce = F.cross_entropy(main_outputs, torch.cat((cutmix_label, label_batch), dim=0))
            loss_main_dice = DICE(main_outputs, torch.cat((cutmix_label, label_batch), dim=0))
            loss_main = (loss_main_ce + loss_main_dice) / 2
            
            # loss for the mim decoder
            loss_mim = F.l1_loss(volume_batch, mim_outputs, reduction='none')
            loss_mim = args.lambda_ * (loss_mim * mask_region).sum() / (mask_region.sum() + 1e-5)
            
            # loss for the aux decoder
            loss_aux_ce = F.cross_entropy(aux_outputs, label_batch)
            loss_aux_dice = DICE(aux_outputs, label_batch)
            loss_aux = (loss_aux_ce + loss_aux_dice) / 2
            
            loss = loss_main + loss_mim + loss_aux

            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_main: %03f, loss_mim: %03f, loss_aux: %03f'%(iter_num, loss, loss_main, loss_mim, loss_aux))

            if iter_num % 200 == 0:
                model.eval()
                val_dice, maxdice1, max_flag = test(model, model, test_loader, maxdice1, stride_xy=18, stride_z=4)

                save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                save_net_opt(model, optimizer, save_mode_path)
                save_net_opt(model, optimizer, save_best_path)

                model.train()

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()



@torch.no_grad()
def test(net, ema_net, val_loader, maxdice=0, stride_xy=18, stride_z=4): # 18,4/ 16,4
    metrics = test_calculate_metric_LA(net, ema_net, val_loader.dataset)
    val_dice = metrics[0]

    if val_dice > maxdice:
        maxdice = val_dice
        max_flag = True
    else:
        max_flag = False
    logging.info('Evaluation : val_dice: %.4f, val_maxdice: %.4f' % (val_dice, maxdice))
    return val_dice, maxdice, max_flag



'''
@torch.no_grad()
def test(net, ema_net, image_list, maxdice=0, stride_xy=16, stride_z=4):
    metrics = test_calculate_metric(net, ema_net, image_list, stride_xy=stride_xy, stride_z=stride_z)
    val_dice = metrics[0]

    if val_dice > maxdice:
        maxdice = val_dice
        max_flag = True
    else:
        max_flag = False
    logging.info('Evaluation : val_dice: %.4f, val_maxdice: %.4f' % (val_dice, maxdice))
    return val_dice, maxdice, max_flag
'''


def self_train(args, pre_snapshot_path, self_snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    
    model = nn.DataParallel(model)
    model = model.cuda()
    
            
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)
    
    
    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    load_net(model, pretrained_model)
    
    
    load_net(model, '/data/userdisk1/qjzeng/semi_seg/IJCAI24/weight/LA_IJCAI_16_labeled_0.2/self_train/VNet_best_model.pth')
    #load_net(ema_model, '/data/userdisk1/qjzeng/semi_seg/IJCAI24/weight/LA_IJCAI_16_labeled_0.2/pre_train/VNet_best_model.pth')
    maxdice1 = 0.
    print(time.localtime(time.time()))
    val_dice, maxdice1, max_flag = test(model, model, test_loader, maxdice1, stride_xy=18, stride_z=4)
    print(time.localtime(time.time()))
    exit()
    
    model.train()
    writer = SummaryWriter(self_snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    maxdice1 = 0.
    max_epoch = self_max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        for _, (sampled_batch, mim_mask) in enumerate(trainloader):
        
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            

            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            unimg = volume_batch[args.labeled_bs:]

            # generate pseudo-label using main decoder
            with torch.no_grad():
                un_main_outputs = model(unimg)
                un_main_outputs = un_main_outputs.detach()
                un_main_prob     = F.softmax(un_main_outputs, dim=1)
                threshold = 0.5
                un_main_ps_lab   = un_main_prob[:,1,:,:,:] > threshold
                un_main_ps_lab   = un_main_ps_lab.unsqueeze(1).detach().int()
                
                un_pl = torch.argmax(un_main_prob, dim=1)


            #----------------------learn a main decoder-------------------------#

            # cutmix for main decoder
            with torch.no_grad():
                img_mask, _ = context_mask(img_a, args.mask_ratio)

            cutmix_batch = img_a * img_mask + img_b * (1 - img_mask)
            cutmix_label = lab_a * img_mask + lab_b * (1 - img_mask)
            
            # forward of main decoder
            main_outputs = model(cutmix_batch)


            #----------------------learn a mim decoder-------------------------#
            # masked input for mim decoder
            mask_region = un_main_ps_lab
            mim_batch = unimg * (1 - mask_region)
            
            # forward of mim decoder
            mim_outputs = model.module.mim_forward(mim_batch)           
            
            
            #----------------------learn an aux decoder-------------------------#
            
            # reconstructed input for aux decoder
            mim_outputs_ = mim_outputs.detach()
            re_batch = unimg * (1 - mask_region) + mim_outputs_.detach() * mask_region
            
            # forward of aux decoder
            aux_outputs = model.module.aux_forward(re_batch)             
            
            
            
            #----------------------loss calculation-------------------------#
            # loss for the main decoder
            loss_main_ce = F.cross_entropy(main_outputs, cutmix_label)
            loss_main_dice = DICE(main_outputs, cutmix_label)
            loss_main = (loss_main_ce + loss_main_dice) / 2
            
            # loss for the mim decoder
            loss_mim = F.l1_loss(unimg, mim_outputs, reduction='none')
            loss_mim = args.lambda_ * (loss_mim * mask_region).sum() / (mask_region.sum() + 1e-5)
            
            # loss for the aux decoder
            loss_aux_ce = F.cross_entropy(aux_outputs, un_pl)
            loss_aux_dice = DICE(aux_outputs, un_pl)
            loss_aux = (loss_aux_ce + loss_aux_dice) / 2
            
            loss = loss_main + loss_mim + loss_aux


            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_main: %03f, loss_mim: %03f, loss_aux: %03f'%(iter_num, loss, loss_main, loss_mim, loss_aux))


             # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 200 == 0:
                model.eval()
                
                val_dice, maxdice1, max_flag = test(model, model, test_loader, maxdice1, stride_xy=18, stride_z=4)

                save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                save_net_opt(model, optimizer, save_mode_path)
                save_net_opt(model, optimizer, save_best_path)
                
                model.train()
            

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "./weight/LA_{}_{}_labeled_{}/pre_train".format(args.exp, args.labelnum, args.lambda_)
    self_snapshot_path = "./weight/LA_{}_{}_labeled_{}/self_train".format(args.exp, args.labelnum, args.lambda_)
    print("Strating Model training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    
    '''
    # -- Pre-Training
    logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)
    '''
    
    
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
    
    

    