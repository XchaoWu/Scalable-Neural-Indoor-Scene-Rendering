from .vgg import VGGLoss
from .ssim import SSIM
import torch 
import torch.nn as nn 
import numpy as np 
import sys 
sys.path.append('../')
import utils 

class Criterion:
    def __init__(self, cfg, device):
        
        self.cfg = cfg 
        self.device = device 

        self.loss_dict = {}

        if self.cfg.MSE_WEIGHT > 0:
            self.MSE = torch.nn.MSELoss()
            self.loss_dict['mse_loss'] = list()
        
        if self.cfg.VGG_WEIGHT > 0:
            self.VGG = VGGLoss().to(self.device)
            self.loss_dict['vgg_loss'] = list()
        
        if self.cfg.SSIM_WEIGHT > 0:
            self.SSIMLOSS = SSIM().to(self.device) 
            self.loss_dict['ssim_loss'] = list()

        if self.cfg.HARD_MINING_WEIGHT > 0:
            self.loss_dict['hard_loss'] = list()

        if self.cfg.DIFFUSE_WEIGHT > 0:
            self.loss_dict['diffuse_loss'] = list()

        if self.cfg.K0_WEIGHT > 0:
            self.loss_dict['k0_loss'] = list()
        
        if self.cfg.KN_WEIGHT > 0:
            self.loss_dict['kn_loss'] = list()

        self.loss_dict['total_loss'] = list()
    
    def record_epoch_loss(self, epoch, writer):
        
        info = ""
        for key in self.loss_dict.keys():
            if len(self.loss_dict[key]) > 0:
                epoch_loss = np.mean(self.loss_dict[key])
                info += f"{key} {epoch_loss:.8f}\n"
                writer.add_scalar(f"{key}", epoch_loss, epoch)
                self.loss_dict[key] = list()
        return info 

    def compute_loss(self, **kwargs):

        loss = 0 

        if  self.cfg.MSE_WEIGHT > 0:
            # mseloss = utils.Mask_MSELoss(kwargs['pred'], kwargs['gtcolor'], kwargs['mask'])
            mseloss = self.MSE(kwargs['pred']*kwargs['mask'], kwargs['gtcolor']*kwargs['mask'])
            self.loss_dict['mse_loss'].append(mseloss.item())
            loss = loss + self.cfg.MSE_WEIGHT * mseloss
        if  self.cfg.VGG_WEIGHT > 0:
            vggloss = self.VGG((kwargs['pred'] * kwargs['mask']).permute(0,3,1,2), 
                            (kwargs['gtcolor'] * kwargs['mask']).permute(0,3,1,2), self.cfg.VGG_LAYER).mean()
            self.loss_dict['vgg_loss'].append(vggloss.item())
            loss = loss + self.cfg.VGG_WEIGHT * vggloss
        if  self.cfg.SSIM_WEIGHT > 0:
            item = torch.sum(kwargs['mask'])
            # if item != 0:
            ssim = self.SSIMLOSS((kwargs['pred'] * kwargs['mask']).permute(0,3,1,2),
                                        (kwargs['gtcolor'] * kwargs['mask']).permute(0,3,1,2)).mean()
            ssimloss = 1.0 - ssim 
            self.loss_dict['ssim_loss'].append(ssimloss.item())
            loss = loss + self.cfg.SSIM_WEIGHT * ssimloss
            # else:
            #     self.loss_dict['ssim_loss'].append(0)

        if  self.cfg.HARD_MINING_WEIGHT > 0 and kwargs['epoch'] > self.cfg.HARDMINING_EPOCH:
            hard_loss = utils.hard_mining_loss(
                            (kwargs['pred'] * kwargs['mask']).permute(0,3,1,2),
                            (kwargs['gtcolor'] * kwargs['mask']).permute(0,3,1,2), ratio=0.1)
            self.loss_dict['hard_loss'].append(hard_loss.item())
            loss = loss + self.cfg.HARD_MINING_WEIGHT * hard_loss
        if self.cfg.DIFFUSE_WEIGHT > 0:
            # diffuse_loss = utils.Mask_MSELoss(kwargs['diffuse_intile'], kwargs['gt_diffuse'], kwargs['mask'])

            diffuse_mask = (kwargs['gt_diffuse'] != 0).float().detach() # [FIXME]
            
            diffuse_loss = torch.nn.MSELoss()(kwargs['diffuse_intile']*diffuse_mask*kwargs['mask'], 
                                              kwargs['gt_diffuse']*diffuse_mask*kwargs['mask'])
            self.loss_dict['diffuse_loss'].append(diffuse_loss.item())
            loss = loss + self.cfg.DIFFUSE_WEIGHT * diffuse_loss   

        if self.cfg.K0_WEIGHT > 0 and kwargs['epoch'] > self.cfg.VIEWDEPENDENT_EPOCH:
            K0 = kwargs['coeffi'][..., 0]
            K0_loss = torch.mean(K0**2)
            self.loss_dict['k0_loss'].append(K0_loss.item())
            loss = loss + self.cfg.K0_WEIGHT * K0_loss    

        if self.cfg.KN_WEIGHT > 0 and kwargs['epoch'] > self.cfg.VIEWDEPENDENT_EPOCH:
            KN = kwargs['coeffi'][..., 1:]
            KN_loss = torch.mean(KN**2)
            self.loss_dict['kn_loss'].append(KN_loss.item())
            loss = loss + self.cfg.KN_WEIGHT * KN_loss
        
        self.loss_dict['total_loss'].append(loss.item())
        
        return loss 