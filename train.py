import argparse
import yaml
import os
from tqdm import tqdm
import torch.nn.functional as F
import warnings
import math
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")
num_dropout_ensembles = 8
parser = argparse.ArgumentParser(description='Input a config file.')
parser.add_argument('--config', help='Config file path')
args = parser.parse_args()
f = open(args.config)
config = yaml.load(f, Loader=yaml.FullLoader)

os.environ["CUDA_VISIBLE_DEVICES"] = '4' # 1:kernel5 3：kernel11 4:step2_centernorm

import numpy as np
from torch.utils.data import DataLoader
from dataset2 import TrainDataset
from dataset import TestDataset
import models
import torch
import loss

#if config['use_wandb']:
 #   import wandb
  #  wandb.init(project=config['project'], entity=config['entity'], name=config['run_name'])

os.makedirs(config['checkpoint_folder'], exist_ok=True)
model = torch.nn.DataParallel(models.mana(config,is_training=True)).cuda()

if config['hot_start']:
    checkpt=torch.load(config['hot_start_checkpt'])
    model.module.load_state_dict(checkpt)

loss_fn = loss.lossfun()

writer = SummaryWriter(log_dir=config['checkpoint_folder'])

train_dataset = TrainDataset(config['dataset_path'], patch_size=config['patch_size'], scale=3)
dataloader = DataLoader(dataset=train_dataset,
                        batch_size=2,
                        shuffle=True,
                        num_workers=config['num_workers'],
                        pin_memory=True)
test_dataset = TestDataset('Lifeact_valid_step2_VSR_centernorm.h5', patch_size=config['patch_size'], scale=3)
test_dataloader = DataLoader(dataset=test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=config['num_workers'],
                        pin_memory=True)
'''
test_dataset = TrainDataset('Micro_test.h5', patch_size=config['patch_size'], scale=3)
testloader = DataLoader(dataset=test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=config['num_workers'],
                        pin_memory=True)

'''
def calc_psnr(sr, hr, scale=3, rgb_range=255, dataset=None):
    #if hr.nelement() == 1: return 0
    #diff = (sr - hr) / rgb_range
    diff = (sr - hr)
    diff = diff.cpu().detach().numpy()
    mse = np.mean((diff) ** 2)
    #mse = valid.pow(2).mean()

    return -10 * math.log10(mse)
count = 0
stage1=config['stage1']
stage2=config['stage2']
stage3=90000
best_valid = 100
for epoch in range(0, config['epoch']):
    loss_list=[]
    loss_list2=[]
    test_list=[]
    psnr_list=[]
    qloss_list=[]
    valid_list = []
    model.train()
    with tqdm(dataloader, desc="Training MANA") as tepoch:
        for inp, gt in tepoch:
            tepoch.set_description(f"Training MANA--Epoch {epoch}")
            if count==0:
                for p in model.module.parameters():
                    p.requires_grad=True
                for p in model.module.nonloc_spatial.W_z1.parameters():
                    p.requires_grad=False
                model.module.nonloc_spatial.mb.requires_grad=False
                #model.module.nonloc_spatial.D.requires_grad=False
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=5e-5, betas=(0.5, 0.999))
                #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1200, eta_min=0, last_epoch=-1)

            elif count==stage1:
                for p in model.module.parameters():
                    p.requires_grad=False
                for p in model.module.nonloc_spatial.W_z1.parameters():
                    p.requires_grad=True
                model.module.nonloc_spatial.mb.requires_grad=True
                #model.module.nonloc_spatial.D.requires_grad=False
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=2e-4, betas=(0.5, 0.999))
                #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=0, last_epoch=-1)
            elif count==stage2:
                for p in model.module.parameters():
                    p.requires_grad=True
                model.module.nonloc_spatial.mb.requires_grad=True
                #model.module.nonloc_spatial.D.requires_grad=False
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=5e-5, betas=(0.5, 0.999))
                #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=0, last_epoch=-1)
            elif count==stage3:
                for p in model.module.parameters():
                    p.requires_grad=True
                model.module.nonloc_spatial.mb.requires_grad=True
                #model.module.nonloc_spatial.D.requires_grad=False
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=5e-5, betas=(0.5, 0.999))
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1)
    
    
            inp = inp.float().cuda()
            gt = gt.float().cuda()
            
            optimizer.zero_grad()
            oup,qloss = model(inp)
            
            if count<stage1:
                loss,loss2 = loss_fn(gt, oup)
                #loss_numpy = loss.cpu().data.numpy()
                loss = loss.mean()
                loss.backward()
                loss_list.append(loss.data.cpu())
                loss_list2.append(loss2.data.cpu())
                optimizer.step()
                tepoch.set_postfix({'L1 Loss': loss.data.cpu().numpy(),'L2 Loss': loss2.data.cpu().numpy(), 'Current Iteration': count, 'Current Stage': 'stage1'})
                #if config['use_wandb']:
                 #   wandb.log({"L1 Loss": loss})
                
                    
            elif count<stage2:
                loss=torch.mean(qloss)
                loss.backward()
                qloss_list.append(loss.data.cpu())
                loss1,loss2 = loss_fn(gt, oup)
                loss1 = loss1.mean()
                loss_list.append(loss1.data.cpu())
                loss_list2.append(loss2.data.cpu())
                optimizer.step()
                tepoch.set_postfix({'Quantize Loss:': loss1.data.cpu().numpy(),'L2 Loss': loss2.data.cpu().numpy(), 'Current Iteration': count, 'Current Stage': 'stage2'})
                #if config['use_wandb']:
                 #   wandb.log({"Quantize Loss": loss})
                
            else:
                loss,loss2 = loss_fn(gt, oup)
                loss = loss.mean()
                loss.backward()
                loss_list.append(loss.data.cpu())
                loss_list2.append(loss2.data.cpu())
                optimizer.step()
                tepoch.set_postfix({'L1 Loss': loss.data.cpu().numpy(),'L2 Loss': loss2.data.cpu().numpy(), 'Current Iteration': count, 'Current Stage': 'stage3'})
                #if config['use_wandb']:
                  #  wandb.log({"L1 Loss": loss})

            count += 1
    
            if count % config['N_save_checkpt'] == 0:
                tepoch.set_description("Training MANA--Saving Checkpoint")
                torch.save(model.module.state_dict(), config['checkpoint_folder'] +'/' + config['checkpoint_name'])
                with torch.no_grad():
                    model.eval()
                    with tqdm(test_dataloader, desc="Valid MANA") as tepoch:
                        for inp, gt in tepoch:
                            model.eval()
                            inp = inp.float().cuda()
                            gt = gt.float().cuda()
                            optimizer.zero_grad()
                            oup,qloss = model(inp)
                            loss,loss2 = loss_fn(gt, oup[:,:,:,:])
                            loss = loss.mean()
                            valid_list.append(loss2.data.cpu())
                writer.add_scalar('Valid/loss', torch.mean(torch.stack(valid_list)), count / config['N_save_checkpt'])
                if torch.mean(torch.stack(valid_list)) < best_valid:
                    best_valid = torch.mean(torch.stack(valid_list))
                    torch.save(model.module.state_dict(), config['checkpoint_folder'] +'/' + 'checkptbest.pt')
                
    writer.add_scalar('Train/loss', torch.mean(torch.stack(loss_list)), epoch)
    if count < stage2 and count > stage1:
        writer.add_scalar('Train/qloss', torch.mean(torch.stack(qloss_list)), epoch)
    writer.add_scalar('Train/loss2', torch.mean(torch.stack(loss_list2)), epoch)
    if count > stage3:
        scheduler.step()
        writer.add_scalar('Train/lr', scheduler.get_last_lr()[0], epoch)
writer.close()