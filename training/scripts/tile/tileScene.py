import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from glob import glob 
import os,cv2 
import datetime, time 
import tools 
import utils 
from .tileGroup import TileGroup
from multiprocess import Process

"""
Scene 
Group Group Group ...
tile tile tile tile ....
"""
class TileScene:
    def __init__(self, cfg, sec_itr = False, cfg_path=None):

        self.cfg = cfg 

        self.sec_itr = sec_itr

        self.cfg.Ks, self.cfg.C2Ws = tools.read_campara(self.cfg.CAMPATH)

        self.cfg.ignore = utils.load_ignore(self.cfg.IGNORE, self.cfg.Ks.shape[0])

        _, self.cfg.centers, self.cfg.IndexMap, self.cfg.SparseToDense, self.cfg.BConFaceIdx, \
        self.cfg.BConFaceNum, self.cfg.VisImg, self.cfg.scene_min_corner, self.cfg.tile_shape = \
        tools.load_preprocess(self.cfg.DATADIR)

        self.cfg.tile_size = self.cfg.NUM_VOXEL * self.cfg.VOXEL_SIZE
        self.cfg.scene_size = self.cfg.tile_shape * self.cfg.tile_size
        self.cfg.far = np.linalg.norm(self.cfg.scene_size)

        self.cfg_path = cfg_path 

    def train_tiles(self, **kwargs):
        try:
            tileIdx_list = kwargs['tileIdx_list']
        except:
            print("plz specify tileIdx_list for training")

        try:
            gpu = kwargs['gpu']
        except:
            gpu = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu}'
        device = torch.device('cuda:0')

        try:
            prefix = kwargs["prefix"]
        except:
            prefix = ""

        if self.cfg.DEBUG:
            self.logdir = os.path.join(self.cfg.LOGDIR, "DEBUG")
        else:
            runtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            if self.cfg.PREFIX != "":
                self.cfg.PREFIX = self.cfg.PREFIX + "-"
            self.logdir = os.path.join(self.cfg.LOGDIR, f"{self.cfg.PREFIX}{prefix}-{runtime}")

        if os.path.exists(self.logdir) is False:
            os.mkdir(self.logdir)
        print(f"\n====== logdir dir {self.logdir} ======\n")

        if self.cfg_path:
            os.system(f"cp {self.cfg_path} {self.logdir}")

        TG = TileGroup(self.cfg, tileIdx_list, self.logdir, device, group_name=prefix, sec_itr=self.sec_itr)
        TG.train()

    def parallel_train(self, **kwargs):

        if self.cfg.DEBUG:
            self.logdir = os.path.join(self.cfg.LOGDIR, "DEBUG")
        else:
            runtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            self.logdir = os.path.join(self.cfg.LOGDIR, f"{self.cfg.PREFIX}{runtime}")

        if os.path.exists(self.logdir) is False:
            os.mkdir(self.logdir)
        print(f"\n====== logdir dir {self.logdir} ======\n")

        if self.cfg_path:
            os.system(f"cp {self.cfg_path} {self.logdir}")

        numGPU = len(self.cfg.GPUIDXS)

        self.group_info = utils.build_training_info(self.cfg.TILEGROUPDIR, numGPU, inverse=False)
 
        print(f"\n====== Find {len(self.group_info)} groups in scene ======\n")

        def unit(start, end, GPUIDX):
            os.environ["CUDA_VISIBLE_DEVICES"] = f'{GPUIDX}'
            device = torch.device('cuda:0')
            for i in range(start, end):
                file, tileIdx_list = self.group_info[i]
                group_name = os.path.splitext(os.path.basename(file))[0]
                path = os.path.join(self.logdir, group_name)
                TG = TileGroup(self.cfg, tileIdx_list, path, device, group_name=group_name,sec_itr=self.sec_itr)
                TG.train()

        base_num = len(self.group_info) // numGPU 
        add_num = len(self.group_info) % numGPU

        start = 0
        p_list = []
        for i,gpu_id in enumerate(self.cfg.GPUIDXS):
            if i < add_num:
                end = start + base_num + 1 
            else:
                end = start + base_num 
            p = Process(target=unit, args=(start, end, gpu_id))
            p_list.append(p)
            start = end 
        
        for p in p_list:
            p.start()

        for p in p_list:
            p.join()
        
        print("\n====== Finished training ======\n")

        

