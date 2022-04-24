from math import gamma
import torch 
import torch.nn as nn
from torch.nn import parameter
from torch.nn import modules 
import torch.nn.functional as F 
from tensorboardX import SummaryWriter
import numpy as np 
from glob import glob 
import os,cv2 
import datetime, time 
import camera, networks
import tools 
import utils 
import random 
from tqdm import tqdm 
import sys 
sys.path.append('../')
from src import preparedata_patch, preparedata_patch_sec
from .tile import Tile 
from loss.criterion import Criterion

class TileGroup:
    def __init__(self, cfg, tileIdx_list, logdir, device, group_name,
                 sec_itr = False, fix_diffuse=False, fix_reflection=False,
                 optim='Adam', scheduler='stepLR'):
        
        self.cfg = cfg 

        self.group_name = group_name

        self.error_tiles = []

        self.logdir = logdir
        self.tileIdx_list = tileIdx_list
        self.tileIdx_list.sort()
        self.tileIdx_list = np.array(self.tileIdx_list)

        self.sec_itr = sec_itr

        self.voxel_size = self.cfg.VOXEL_SIZE 
        self.num_voxel = self.cfg.NUM_VOXEL 
        self.tile_size = self.voxel_size * self.num_voxel
        self.num_tile = len(self.tileIdx_list)
        self.nsamples = self.cfg.NSAMPLES

        try:
            self.Ks = self.cfg.Ks 
            self.C2Ws = self.cfg.C2Ws 
        except:
            self.Ks, self.C2Ws = tools.read_campara(self.cfg.CAMPATH)

        if os.path.exists(logdir) is False:
            os.mkdir(logdir)

        self.device = device

        self.cfg.NUM_BASIS = (self.cfg.DEG+1) ** 2

        # self.debug_grad()

        self.tile_centers = self.cfg.centers[self.tileIdx_list].copy()

        func = lambda x: torch.from_numpy(x).to(self.device)
        group_center = np.mean(self.tile_centers, axis=0)
        self.group_center = func(group_center)
        self.group_min_corner = func(np.min(self.tile_centers,axis=0) - self.cfg.tile_size / 2.)
        self.group_max_corner = func(np.max(self.tile_centers,axis=0) + self.cfg.tile_size / 2.)
        self.group_size = self.group_max_corner - self.group_min_corner
        self.scene_size = func(self.cfg.scene_size).float()

        np.save(os.path.join(self.logdir, "group_center.npy"), group_center)

        self.norm_diffuse_point = lambda x: (x - self.group_center) / self.group_size
        self.norm_specular_point = lambda x: (x - self.group_center) / self.cfg.far


        self.tiles = []
        for tileIdx in self.tileIdx_list:
            t = Tile(self.cfg, tileIdx, self.cfg.centers[tileIdx], self.logdir, self.device,
                    self.norm_diffuse_point, self.norm_specular_point)
            self.tiles.append(t)

        self.build_shared_MLP(optim=optim, scheduler=scheduler,
                              fix_diffuse=fix_diffuse, fix_reflection=fix_reflection)
        


        if os.path.isfile(os.path.join(self.cfg.WORKSPACE, "predefine.txt")):
            self.predefine = []
            with open(os.path.join(self.cfg.WORKSPACE, "predefine.txt"), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(" ")
                    if self.group_name == line[0]:
                        self.predefine = [int(item) for item in line[1:]]
                        print("find predefine imgIdxs")
                        break 
        else:
            self.predefine = []
        self.predefine = np.array(self.predefine)

        # self.debug_grad()

        self.stop_training = False
    
    def build_training_context(self):
        self.train_error = False 

        self.pruning_step = 0

        self.prepare_training_data()

        for idx in range(self.num_tile):
            if self.tileIdx_list[idx] not in [item[0] for item in self.error_tiles]:
                if self.sec_itr or self.cfg.INIT_VOXEL == False:
                    self.tiles[idx].build_training_context(None)
                else:
                    self.tiles[idx].build_training_context(self.init_nodes_flag[idx])

        self.writer = SummaryWriter(self.logdir)

        with open(os.path.join(self.logdir, 'errorTiles.txt'), 'a') as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            for item in self.error_tiles:
                f.write(f"Time {current_time}\ttileIdx: {item[0]}\tError: {item[1]}\n")
        
        # self.infer_gif_all(0)
        # exit()
    
    
    def prepare_training_data(self):

        
        if self.sec_itr:
            data_func = preparedata_patch_sec
            # compute_tileIdxs = self.tileIdx_list[:5]

            try:
                f = open(os.path.join(self.cfg.WORKSPACE, "cross_group.txt"), 'r')
            except:  
                compute_tileIdxs = utils.get_neighborNtiles(self.tileIdx_list.copy(), self.cfg.SparseToDense,
                                            self.cfg.IndexMap, self.cfg.tile_shape, dilate_size=self.cfg.EXPAND_TILES)
                print(f"auto expand {self.cfg.EXPAND_TILES} tiles")
            else:
                lines = f.readlines()

                compute_tileIdxs = []
                for line in lines:
                    line = line.strip().split(' ')
                    if self.group_name == line[0]:
                        group_name_list = line[1:]
                        compute_tileIdxs = utils.search_tiles_by_groups(self.cfg.PRETRAINED, group_name_list)
                        break
                f.close()

                if len(compute_tileIdxs) == 0:
                    compute_tileIdxs = utils.get_neighborNtiles(self.tileIdx_list.copy(), self.cfg.SparseToDense,
                                                self.cfg.IndexMap, self.cfg.tile_shape, dilate_size=self.cfg.EXPAND_TILES)
                    print(f"auto expand {self.cfg.EXPAND_TILES} tiles")
                else:
                    print(f"cross group {group_name_list}")

            group_voxels, group_nodes, compute_tileIdxs = utils.search_pretrained_tiles(self.cfg.PRETRAINED, compute_tileIdxs)
            tileIdxInGroup = -1 * np.ones((self.cfg.centers.shape[0]), dtype=np.int32)
            for idx, tileIdx in enumerate(compute_tileIdxs):
                tileIdxInGroup[tileIdx] = idx 
            
            group_nodes = np.ascontiguousarray(group_nodes, dtype=np.int16)
            group_voxels = np.ascontiguousarray(group_voxels, dtype=np.float32)

            predefine_images = np.ascontiguousarray(np.array(self.cfg.FILTER_IMGS.PREDEFINE))

            # print(predefine_images)

            data, tiles_flag = data_func(self.cfg.IMAGEPATH, self.cfg.MIRROR_DEPTHPATH, 
                                group_voxels.astype(np.float32).flatten(),
                                group_nodes.astype(np.int16).flatten(),
                                tileIdxInGroup.astype(np.int32).flatten(),
                                compute_tileIdxs.astype(np.int32).flatten(),
                                self.cfg.VisImg.flatten().astype(np.int32),
                                self.cfg.ignore.flatten().astype(np.int32),
                                self.cfg.Ks, self.cfg.C2Ws,
                                self.cfg.centers.flatten().astype(np.float32),
                                self.cfg.IndexMap.flatten().astype(np.int32), 
                                predefine_images.flatten().astype(np.int32),
                                self.cfg.HEIGHT, self.cfg.WIDTH, 
                                self.cfg.PATCH_SIZE, self.cfg.MAX_TRACING_TILE,
                                *list(self.cfg.tile_shape), *list(self.cfg.scene_min_corner), 
                                self.cfg.NUM_VOXEL+2, self.cfg.VOXEL_SIZE,    
                                self.cfg.FILTER_IMGS.NUM_TILES*self.tile_size,
                                self.cfg.VOXEL_SIZE * 0.5, 
                                self.cfg.FILTER_IMGS.NEED, False)

        else:
            if len(self.predefine) == 0:
                need = self.cfg.FILTER_IMGS.NEED
            else:
                need = 0
            data_func = preparedata_patch
            data, tiles_flag, init_nodes_flag = \
                data_func(self.cfg.PATH, self.cfg.IMAGEPATH, self.cfg.DEPTHPATH,
                        self.tileIdx_list.flatten().astype(np.int32),
                        self.cfg.VisImg.flatten().astype(np.int32),
                        self.cfg.ignore.flatten().astype(np.int32),
                        self.cfg.Ks, self.cfg.C2Ws,
                        self.cfg.centers.flatten().astype(np.float32),
                        self.cfg.IndexMap.flatten().astype(np.int32),
                        self.cfg.BConFaceIdx.flatten().astype(np.int32),
                        self.cfg.BConFaceNum.flatten().astype(np.int32),
                        self.predefine.flatten().astype(np.int32),
                        self.cfg.HEIGHT, self.cfg.WIDTH, self.cfg.NUM_BLOCK,
                        self.cfg.PATCH_SIZE, self.cfg.MAX_TRACING_TILE,
                        *list(self.cfg.tile_shape), *list(self.cfg.scene_min_corner), 
                        self.cfg.NUM_VOXEL, self.cfg.VOXEL_SIZE, 
                        self.cfg.VOXEL_SIZE*self.cfg.DILATE_VOXELS, 
                        self.cfg.FILTER_IMGS.NUM_TILES*self.tile_size, need, False)

        try:
            if self.sec_itr:
                data = np.frombuffer(data, dtype=np.float32).reshape(-1,self.cfg.PATCH_SIZE**2,14)
            else:
                data = np.frombuffer(data, dtype=np.float32).reshape(-1,self.cfg.PATCH_SIZE**2,14)
        except:
            print("\n====== No training data ======\n")
            self.train_error = True
            self.error_tiles = [(item, "No training Data") for item in self.tileIdx_list]
            return 
        else:
            print(data.shape)
            if self.sec_itr:
                mask = data[...,13:14]

                diff = np.abs(data[..., 7:10] - data[..., 10:13])
                diff = np.sum(diff * mask, axis=1) / np.sum(mask, axis=1)

                # diff = np.mean(np.abs(data[..., 7:10] - data[..., 10:13]), axis=1)
                mean_diff = np.mean(diff, axis=-1)
                # print(mean_diff)
                # exit()
                # hard_mask = mean_diff > 0.05

                rank = np.argsort(mean_diff)[::-1]
                data = data[rank[:self.cfg.FILTER_IMGS.PATCH_NEED]]

            if data.shape[0] == 0:
                print("\n====== No training data ======\n")
                self.train_error = True
                self.error_tiles = [(item, "No training Data") for item in self.tileIdx_list]
                return 
            print(data.shape)
            self.trainData = torch.from_numpy(data.copy()).float()
            self.len_Data = self.trainData.shape[0]
        
        try:
            tiles_flag = np.frombuffer(tiles_flag, dtype=np.int32).reshape(-1)
        except:
            print("\n====== No training data ======\n")
        else:
            print(tiles_flag.shape)
            self.tiles_flag = tiles_flag
            non_empty_tiles = list(set(list(self.tiles_flag)))
            for idx in range(self.num_tile):
                if idx not in non_empty_tiles:
                    self.error_tiles.append((self.tileIdx_list[idx], "No training Data"))
        
        if self.sec_itr == False and self.cfg.INIT_VOXEL:
            try:
                init_nodes_flag = np.frombuffer(init_nodes_flag, dtype=np.int16).reshape(self.num_tile, self.num_voxel,self.num_voxel,self.num_voxel)
            except:
                print("\n====== init_nodes error ======\n")
            else:
                print(init_nodes_flag.shape)
                # NT x NV x NV x NV 
                self.init_nodes_flag = torch.from_numpy(init_nodes_flag.copy()).permute(0,3,2,1).contiguous()

        self.MEM_DATA = (data.size * 4) / (1000**3)
        print(f"\n====== Training data mem: {self.MEM_DATA} GB ======\n")
            
    def debug_grad(self):
        def hook_fn_backward(module, grad_input, grad_output):
            print(module)
            abs_grad = torch.abs(grad_input[0])
            mean_grad = torch.mean(abs_grad)
            min_grad = torch.min(abs_grad)
            max_grad = torch.max(abs_grad)
            print(f"Input:\nmean grad {mean_grad:.8f}\nmin grad {min_grad:.8f}\nmax grad {max_grad:.8f}\n")
            abs_grad = torch.abs(grad_output[0])
            mean_grad = torch.mean(abs_grad)
            min_grad = torch.min(abs_grad)
            max_grad = torch.max(abs_grad)
            print(f"Output:\nmean grad {mean_grad:.8f}\nmin grad {min_grad:.8f}\nmax grad {max_grad:.8f}\n")
        for name, module in self.coeffi_model.named_children():
            module.register_backward_hook(hook_fn_backward)
    
    
    def build_shared_MLP(self, init_mode='kaiming', optim='Adam', scheduler='expLR',
                         fix_diffuse=False, fix_reflection=False, ):

        self.dicts = {}
        params = []

        if self.sec_itr == False:
            model = networks.MLP(hiden_depth=self.cfg.SURFACE_MLP.DEPTH,
                                hiden_width=self.cfg.SURFACE_MLP.WIDTH,
                                in_channel=3,
                                L=self.cfg.SURFACE_MLP.L,
                                num_out=4,
                                activation=torch.nn.ReLU())
            model = networks.init_model(model, mode=init_mode)
            self.model = model.to(self.device)
            self.dicts["model"] = self.model 
            if fix_diffuse == False:
                params.append({"params":self.model.parameters(), "lr": self.cfg.ETA})

        coeffi_model = networks.MLP(hiden_depth=self.cfg.REFLECTION_MLP.DEPTH,
                                    hiden_width=self.cfg.REFLECTION_MLP.WIDTH,
                                    in_channel=3,
                                    L=self.cfg.REFLECTION_MLP.L,
                                    num_out=self.cfg.NUM_BASIS*3+1,
                                    activation=torch.nn.LeakyReLU(0.01))
        coeffi_model = networks.init_model(coeffi_model, mode=init_mode)
        self.coeffi_model = coeffi_model.to(self.device)
        self.dicts["coeffi"] = self.coeffi_model

        if fix_reflection == False:
            params.append({"params": self.coeffi_model.parameters(), "lr": self.cfg.ETA_REFLECTION})

        if self.cfg.PRETRAINED != '':
            path = self.cfg.PRETRAINED
            if os.path.isdir(path) is False:
                print("\n====== pretrained path is not avaliable! ======\n")
                return False
            info = utils.get_pretrained_tiles(os.path.join(path,self.group_name))
            success_load = self.load_pretrained(info)
            if success_load == False:
                print("Load pretrained failed!")
            else:
                print(f"Load pretrained from {self.cfg.PRETRAINED}")

        if optim == 'Adam':
            self.optimizer = torch.optim.Adam(params, weight_decay=1e-5)
            # self.optimizer = torch.optim.SGD(params, weight_decay=1e-5)
            self.dicts['optimizer'] = self.optimizer
        else:
            print('no optimizer for group')
            pass
        
        if scheduler == 'stepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                             step_size=self.cfg.SCHDULER.STEP_SIZE, 
                                                             gamma=self.cfg.SCHDULER.GAMMA)
            self.dicts['scheduler'] = self.scheduler
        elif scheduler == 'expLR':
            self.scheduler =  torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 
                                                                     gamma=0.98)
        else:
            print('no scheduler for group')
            pass
        
        self.criterions = Criterion(self.cfg, self.device)
        self.dicts['criterions'] = self.criterions

        print("\n====== Finished building model ======\n")
    
    def fine_tune_prepare(self,):
        """
        for fine tune step 
        """
        if self.sec_itr:
            return
        
        eta = utils.get_lr(self.optimizer)
        for idx in range(self.num_tile):
            if self.tileIdx_list[idx] not in [item[0] for item in self.error_tiles]:
                self.optimizer.add_param_group({'params': self.tiles[idx].voxels, "lr": eta})
        
        self.cfg.DIFFUSE_WEIGHT = 0



    def pruning_all(self, **kwargs):
        if self.sec_itr:
            return
        for idx in range(self.num_tile):
            if self.tileIdx_list[idx] not in [item[0] for item in self.error_tiles]:
                self.tiles[idx].pruning(self.model, 
                                        self.trainDataGPU[self.tiles_flag==idx],
                                        block=self.cfg.PRUNING.SCHEDULE.BLOCK[self.pruning_step],
                                        th=self.cfg.PRUNING.SCHEDULE.THRESH[self.pruning_step],
                                        T_th=self.cfg.PRUNING.SCHEDULE.TRANS[self.pruning_step])
        
    def infer_gif_all(self, epoch, **kwargs):
        for idx in range(self.num_tile):
            if self.tileIdx_list[idx] not in [item[0] for item in self.error_tiles]:
                self.tiles[idx].infer_gif(self.coeffi_model, epoch, fps=15, save_img=False)
                # self.tiles[idx].infer_video(self.coeffi_model, epoch, fps=15, save_img=False)
    
    @torch.no_grad()
    def compute_voxels_all(self):
        if self.sec_itr:
            return 
        for idx in range(self.num_tile):
            if self.tileIdx_list[idx] not in [item[0] for item in self.error_tiles]:
                self.tiles[idx].compute_voxels(self.model)
    
    def check_empty(self):
        count = 0
        for idx in range(self.num_tile):
            if self.tileIdx_list[idx] not in [item[0] for item in self.error_tiles]:
                count += 1
        if count == 0:
            self.stop_training = True

    def train(self, **kwargs):
        
        self.build_training_context()

        if self.train_error:
            return  

        self.trainDataGPU = self.trainData.to(self.device)

        self.index_list = np.arange(self.len_Data)
        

        self.compute_voxels_all()

        print("Start training ...\n")

        self.train_FLAG = 0

        # self.infer_gif_all(0)

        # exit()

        for epoch in range(1, self.cfg.EPOCH+1):

            random.shuffle(self.index_list)

            if epoch >= self.cfg.VIEWDEPENDENT_EPOCH:
                self.train_FLAG = 2

            self.train_one_epoch(epoch, **kwargs)


            self.check_empty()
            if self.stop_training:
                break 

            self.compute_voxels_all()

            if epoch > self.cfg.PRUNING.SCHEDULE.EPOCH[self.pruning_step]:
                self.pruning_step = self.pruning_step + 1
            
            if self.cfg.PRUNING.ENABLE and epoch % self.cfg.EPOCH_PRUNING == 0:
                self.pruning_all()

            if epoch == 1 or epoch % self.cfg.EPOCH_INFER == 0:
            # if epoch % self.cfg.EPOCH_INFER == 0:
                self.infer_gif_all(epoch)

            if epoch % self.cfg.EPOCH_SAVE == 0:
                self.save(epoch)  
  

            self.scheduler.step()  


        print("Finished training")
        tgt_file = os.path.join(self.cfg.RENDERDIR, self.group_name)
        
        relative_path = os.path.join('../logs/', os.path.split(self.logdir)[-1])
        os.system(f"cd {self.cfg.RENDERDIR} && ln -s {relative_path} {self.group_name}") 


    def train_one_epoch(self, epoch, **kwargs):

        forward_time = 0
        backward_time = 0
        epoch_time = 0

        itr = 0

        for i,idx in enumerate(tqdm(self.index_list)):

            batch_data = self.trainDataGPU[idx:idx+1]

            if itr % self.cfg.LOGIMG_STEP == 0:
                log_img = True 
            else:
                log_img = False

            if self.sec_itr:
                out_info,ret = self.tiles[0].train_batch_sec(epoch, batch_data, log_img, **self.dicts)
            else:
                target_tileIdx = self.tiles_flag[idx]
                out_info,ret = self.tiles[target_tileIdx].train_batch(epoch, batch_data, log_img, self.train_FLAG, self.nsamples, **self.dicts)
            
                if ret == False:
                    if self.tiles[target_tileIdx].train_error:
                        if target_tileIdx not in [item[0] for item in self.error_tiles]:
                            self.error_tiles.append((target_tileIdx, self.tiles[target_tileIdx].error_mode))
                    continue 

            forward_time +=  out_info['forward_time']
            backward_time += out_info['backward_time']
            epoch_time += out_info['batch_time']

            itr += 1

        info = f"\n====== Group {self.group_name}\tEpoch {epoch}/{self.cfg.EPOCH} ======\n"
        formal_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        info += f"TIME: {formal_time}\n"

        info += self.criterions.record_epoch_loss(epoch, self.writer)
        forward_time = forward_time / 60
        backward_time = backward_time / 60
        epoch_time = epoch_time / 60 

        eta = utils.get_lr(self.optimizer)
        info += f"nsamples: {self.nsamples}\tleaning rate {eta:.8f}\n"
        info += f"epoch time: {epoch_time:.3f} min\tforward time: {forward_time:.3f} min\tbackward time: {backward_time:.3f} min\n"
        info += f"train_FLAG: {self.train_FLAG}"

        with open(os.path.join(self.logdir, f"TrainLog.txt"), 'a') as f:
            f.write(info)
        
        print(info)

    def save(self, epoch):
        
        for idx in range(self.num_tile):
            if self.tileIdx_list[idx] not in [item[0] for item in self.error_tiles]:
                self.tiles[idx].save(epoch)

        if self.sec_itr == False:
            model_save_path = os.path.join(self.logdir, f"Model-TShared-E{epoch}.pt")
            try:
                torch.save(self.model.state_dict(), model_save_path)
            except:
                print("Model saved failed")
            else:
                print(f"Model saved successfully to {model_save_path}")

        coeffi = os.path.join(self.logdir, f"Coeffi-TShared-E{epoch}.pt")
        try:
            torch.save(self.coeffi_model.state_dict(), coeffi)
        except:
            print("Coeffi model saved failed!")
        else:
            print(f"Coeffi model saved successfully to {coeffi}")
    
    def load_voxel(self, voxels_path_list):
        print('loading voxels')
        for idx, path in tqdm(enumerate(voxels_path_list)):
            self.tiles[idx].load_voxel(path)
    
    def load_nodes_flag(self, nodes_path_list):
        print('loading nodes')
        for idx, path in tqdm(enumerate(nodes_path_list)):
            self.tiles[idx].load_nodes_flag(path)

    def load_pretrained(self, info):
        tileIdx_list = info['tileIdxs']
        if len(self.tileIdx_list) == 0:
            return False 
        coeffi_path = info['coeffi']
        model_path = info['model']
        voxels_path_list = info['voxels']
        nodes_path_list = info['nodes']

        for idx,tileIdx in enumerate(self.tileIdx_list):
            if tileIdx in tileIdx_list:
                idx_in_pretrained = tileIdx_list.index(tileIdx)
                self.tiles[idx].load_voxel(voxels_path_list[idx_in_pretrained])
                self.tiles[idx].load_nodes_flag(nodes_path_list[idx_in_pretrained])
                print(f"loading voxel and node of tile {tileIdx}")
            else:
                self.error_tiles.append((tileIdx, 'no pretrained nodes flag'))
                print(f"tile {tileIdx} not in pretrained dir")

        if self.cfg.LOAD_MODEL and self.sec_itr == False:
            self.model.load_state_dict(torch.load(model_path))
            print(f"load pretrained model from {model_path}")
        if self.cfg.LOAD_COEFF:
            self.coeffi_model.load_state_dict(torch.load(coeffi_path))
            print(f"load pretrained coeffi model from {coeffi_path}")
        return True
    


    
