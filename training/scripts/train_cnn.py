import torch 
import torch.nn as nn 
import numpy as np
import torchvision 
import utils 
import os,sys 
from torch.utils.data import DataLoader
import torch.optim as optim
from cnn.cnn import NeuralSuperSamplingSingle
from loss.vgg import VGGLoss
from  loss.ssim import SSIM
import datetime
import datasets
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
import cv2 

cfg = utils.parse_yaml(sys.argv[1])
runtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

logdir = os.path.join(cfg.LOGDIR, f"{runtime}")
if os.path.exists(logdir) is False:
    os.mkdir(logdir)

gpus = list(cfg.GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in gpus])
print('\nGPU ID {}'.format(' '.join([str(item) for item in gpus])))
gpus = list(range(len(gpus)))


model = NeuralSuperSamplingSingle()
model = nn.DataParallel(model, device_ids=gpus).cuda()
if cfg.PRETRAINED != '':
    model_dict = torch.load(cfg.PRETRAINED)
    model.module.load_state_dict(model_dict)

VGGloss = VGGLoss().cuda()
SSIMLoss = SSIM().cuda()
Grad = utils.Img2Gradient(3).cuda()

optimizer = optim.Adam(model.parameters(), lr=cfg.ETA, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma = 0.95)
print("Finished creating training context\n")

train_dataset = datasets.CNNDataset(cfg, mode='train')
train_loader = DataLoader(train_dataset,batch_size=cfg.BATCHSIZE, 
                            shuffle=True, num_workers=10, pin_memory=False)

val_dataset = datasets.CNNDataset(cfg, mode='val')
val_loader = DataLoader(val_dataset,batch_size=1, 
                            shuffle=True, num_workers=10, pin_memory=False)

writer = SummaryWriter(logdir)


for epoch in range(1, cfg.EPOCH+1):

    if cfg.SSIM_WEIGHT > 0:
        ssim_loss_list = []
    if cfg.VGG_WEIGHT > 0:
        vgg_loss_list = []
    if cfg.MSE_WEIGHT > 0:
        mse_loss_list = []
    if cfg.GRAD_WEIGHT > 0:
        grad_loss_list = []
    loss_list = []
    hard_loss_list = []
    
    for step, batch in enumerate(train_loader, 1):
        global_step = (epoch - 1) * len(train_loader) + step 
        inputs, mask, gt = batch 
        inputs = inputs.cuda()
        mask = mask.cuda()
        boundary = inputs[:,4:5]
        gt = gt.cuda()

        pred = model(inputs)

        loss = 0
        if cfg.MSE_WEIGHT > 0:
            mseloss = torch.nn.MSELoss()(pred*mask, gt*mask) + torch.nn.MSELoss()(pred * boundary * mask, gt * boundary * mask)
            loss = loss + cfg.MSE_WEIGHT * mseloss
            mse_loss_list.append(mseloss.item())
        if cfg.SSIM_WEIGHT > 0:
            ssimloss = SSIMLoss(pred * mask, gt * mask).mean()
            loss = loss + cfg.SSIM_WEIGHT * (1 - ssimloss)
            ssim_loss_list.append(ssimloss.item())
        if cfg.VGG_WEIGHT > 0:
            vggloss = VGGloss(pred * mask, gt * mask, cfg.VGG_LAYER).mean()
            loss = loss + cfg.VGG_WEIGHT * vggloss
            vgg_loss_list.append(vggloss.item())
        if cfg.GRAD_WEIGHT > 0:
            gradloss = torch.nn.MSELoss()(Grad(pred * mask), Grad(gt * mask))
            loss = loss + cfg.GRAD_WEIGHT * gradloss
            grad_loss_list.append(gradloss.item())
        

        if epoch >= 0:
            hard_loss = utils.hard_mining_loss(pred * mask, gt * mask, ratio=0.1)
            loss = loss + 0.1 * hard_loss
            hard_loss_list.append(hard_loss.item())

        loss_list.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    info = f'\n===== epoch {epoch}/{cfg.EPOCH} =====\n'
    if cfg.SSIM_WEIGHT > 0:
        mean_loss = np.mean(ssim_loss_list)
        info += f"ssim: {mean_loss:.6f}\n"
        writer.add_scalar("ssim", mean_loss, epoch)
    if cfg.VGG_WEIGHT > 0:
        mean_loss = np.mean(vgg_loss_list)
        info += f"vgg loss: {mean_loss:.6f}\n"
        writer.add_scalar("vgg loss", mean_loss, epoch)
    if cfg.MSE_WEIGHT > 0:
        mean_loss = np.mean(mse_loss_list)
        info += f"mse loss: {mean_loss:.6f}\n"
        writer.add_scalar("mse loss", mean_loss, epoch)  
    if cfg.GRAD_WEIGHT > 0:
        mean_loss = np.mean(grad_loss_list)
        info += f"grad loss: {mean_loss:.6f}\n"
        writer.add_scalar("grad loss", mean_loss, epoch)  


    if epoch > 0:
        mean_loss = np.mean(hard_loss_list)
        info += f"hard loss: {mean_loss:.6f}\n"
        writer.add_scalar("hard loss", mean_loss, epoch)  

    record_img = torchvision.utils.make_grid(
        torch.cat([inputs[:4,:3,...].clamp(0,1),
                   inputs[:4,4:5,...].clamp(0,1).repeat_interleave(3,dim=1),
                    pred[:4,:3,...].clamp(0,1), 
                    gt[:4, :3].clamp(0,1)]),
        padding=2,
        pad_value=255,
        nrow=4)
    writer.add_image("inputs_pred_gt", record_img, epoch)

    mean_loss = np.mean(loss_list)
    info += f"total loss: {mean_loss:.6f}\n"
    writer.add_scalar("total loss", mean_loss, epoch)
    print(info)

    scheduler.step()

    if epoch % cfg.VAL_EPOCH == 0:

        if cfg.SSIM_WEIGHT > 0:
            ssim_loss_list = []
        if cfg.VGG_WEIGHT > 0:
            vgg_loss_list = []
        psnrs = []

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader, 1):
                inputs, mask, gt = batch 
                inputs = inputs.cuda()
                gt = gt.cuda()
                mask = mask.cuda()
                pred = model(inputs)

                loss = 0
                if cfg.SSIM_WEIGHT > 0:
                    ssimloss = SSIMLoss(pred, gt).mean()
                    loss = loss + cfg.SSIM_WEIGHT * (1 - ssimloss)
                    ssim_loss_list.append(ssimloss.item())
                if cfg.VGG_WEIGHT > 0:
                    vggloss = VGGloss(pred, gt, cfg.VGG_LAYER).mean()
                    loss = loss + cfg.VGG_WEIGHT * vggloss
                    vgg_loss_list.append(vggloss.item())
                psnr = utils.cal_psnr(pred.cpu()*255, gt.cpu()*255)
                psnrs.append(psnr)

            info = f'\n===== VAL epoch {epoch}/{cfg.EPOCH} =====\n'
            if cfg.SSIM_WEIGHT > 0:
                mean_loss = np.mean(ssim_loss_list)
                info += f"val ssim: {mean_loss:.6f}\n"
                writer.add_scalar("val ssim", mean_loss, epoch)
            if cfg.VGG_WEIGHT > 0:
                mean_loss = np.mean(vgg_loss_list)
                info += f"val vgg loss: {mean_loss:.6f}\n"
                writer.add_scalar("val vgg loss", mean_loss, epoch)
            mean_loss = np.mean(psnrs)
            info += f"psnr: {mean_loss:.6f}\n"
            writer.add_scalar("psnr", mean_loss, epoch)
            
            record_img = torchvision.utils.make_grid(
                torch.cat([inputs[:2,:3,...].clamp(0,1),
                            inputs[:2,4:5,...].clamp(0,1).repeat_interleave(3,dim=1),
                            mask[:2].clamp(0,1).repeat_interleave(3,dim=1),
                            pred[:2,:3,...].clamp(0,1), 
                            gt[:2, :3].clamp(0,1)]),
                padding=2,
                pad_value=255,
                nrow=2)
            writer.add_image("val inputs_pred_gt", record_img, epoch)
            print(info)

        model.train()
    
    if epoch % cfg.SAVEEPOCH == 0:
        model_path = os.path.join(logdir,'epoch-{}.pth'.format(epoch))
        torch.save(model.module.state_dict(), model_path)
        print('Model saved to {}'.format(model_path))