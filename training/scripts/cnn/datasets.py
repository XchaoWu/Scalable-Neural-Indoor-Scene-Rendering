import torch 
import torch.utils.data as data
import torch.nn.functional as F
import cv2,os,sys 
from glob import glob 
import numpy as np 
from tqdm import tqdm 
sys.path.append('../')
import utils 

DEBUG_NUM = None

class CNNDataset(data.Dataset):
    def __init__(self, cfg, mode):
        super(CNNDataset, self).__init__()
        if mode not in ['train','val','infer','evaluate']:
            raise ValueError('mode must be one of [train, test, inference]')
        self.cfg = cfg 
        self.mode = mode 
        # LR
        self.img_size = self.cfg.IMG_SIZE
        self.crop_size = self.cfg.CROP_SIZE

        
        self.ignore_list = utils.load_ignore_v2(self.cfg.IGNORE)
        self.data_list = []

        if self.mode == 'train':
            self.data_process = self.data_process_crop
            self.load_data()
        elif self.mode == 'val':
            self.data_process = self.data_process_val
            self.load_val_data()
        else:
            raise NotImplementedError
        print(f"\n===== Finished {mode} data load {self.__len__()} =====\n")

    def load_data(self):
        pred_files = glob(os.path.join(self.cfg.PREDVIEW, "*.png"))
        gt_files = glob(os.path.join(self.cfg.IMAGEPATH, "*.png"))
        # assert len(gt_files) == len(pred_files)
        numImg = len(gt_files)
        if DEBUG_NUM != None:
            numImg = DEBUG_NUM
        
        for file in tqdm(gt_files):
            file_name = os.path.basename(file)
            file_idx = int(os.path.splitext(file_name)[0])
            if file_idx in self.ignore_list:
                continue
            x = cv2.imread(os.path.join(self.cfg.PREDVIEW,f"{file_idx}.png"))
            boundary = cv2.imread(os.path.join(self.cfg.PREDBOUNDARY,f"{file_idx}.png"))[...,0:1]
            mask = cv2.imread(os.path.join(self.cfg.MASK,f"{file_idx}.png"))[...,0:1]
            # diffuse = cv2.imread(os.path.join(self.cfg.PREDDIFFUSE,f"{file_idx}.png"))
            # specular = cv2.imread(os.path.join(self.cfg.PREDSPECULAR,f"{file_idx}.png"))
            gt = cv2.imread(os.path.join(self.cfg.IMAGEPATH,f"{file_idx}.png"))
            inverse_near = np.load(os.path.join(self.cfg.INVERSE_NEAR,f"{file_idx}.npy"))[...,None]
            x = x[...,::-1].copy()
            gt = gt[...,::-1].copy()
            self.data_list.append([x,inverse_near,boundary,mask,gt])

    def load_val_data(self):
        with open(self.cfg.VAL, 'r') as f:
            lines = f.readlines()

        val_list = [item.strip() for item in lines]
        if DEBUG_NUM:
            val_list = val_list[:DEBUG_NUM]
        for i in tqdm(val_list):
            x = cv2.imread(os.path.join(self.cfg.PREDVIEW,f"{i}.png"))
            boundary = cv2.imread(os.path.join(self.cfg.PREDBOUNDARY,f"{i}.png"))[...,0:1]
            mask = cv2.imread(os.path.join(self.cfg.MASK,f"{i}.png"))[...,0:1]
            # diffuse = cv2.imread(os.path.join(self.cfg.PREDDIFFUSE,f"{i}.png"))
            # specular = cv2.imread(os.path.join(self.cfg.PREDSPECULAR,f"{i}.png"))
            gt = cv2.imread(os.path.join(self.cfg.IMAGEPATH,f"{i}.png"))
            inverse_near = np.load(os.path.join(self.cfg.INVERSE_NEAR,f"{i}.npy"))[...,None]
            x = x[...,::-1].copy()
            gt = gt[...,::-1].copy()
            self.data_list.append([x,inverse_near,boundary,mask,gt])


    def __len__(self):
        return len(self.data_list)
    
    def data_process_crop(self, data):
        input_x = data[0]
        # input_specular = data[1]
        input_disparity = data[1]
        input_boundary = data[2]
        input_mask = data[3]
        gt = data[4]

        crop = utils.RandomCrop(self.img_size, self.crop_size)
        crop_input_x = crop(input_x)
        crop_input_disparity = crop(input_disparity)
        crop_input_boundary = crop(input_boundary)
        crop_input_mask = crop(input_mask)
        crop_gt = crop(gt)

        crop_input_x = torch.from_numpy(crop_input_x.transpose(2,0,1)) / 255. 
        crop_input_disparity = torch.from_numpy(crop_input_disparity.transpose(2,0,1))
        crop_input_boundary = torch.from_numpy(crop_input_boundary.transpose(2,0,1)) / 255.
        crop_input_mask = torch.from_numpy(crop_input_mask.transpose(2,0,1)) / 255.
        crop_gt = torch.from_numpy(crop_gt.transpose(2,0,1)) / 255. 
        return torch.cat([crop_input_x,crop_input_disparity, crop_input_boundary], 0).float(), crop_input_mask.float(), crop_gt.float()

    def data_process_val(self, data):
        input_x = data[0]

        H,W = input_x.shape[:2]
        H = H // 4 * 4 
        W = W // 4 * 4
        # input_specular = data[1]
        input_x = input_x[:H,:W]
        input_disparity = data[1][:H,:W]
        input_boundary = data[2][:H,:W]
        input_mask = data[3][:H,:W]
        gt = data[4][:H,:W]

        input_x = torch.from_numpy(input_x.transpose(2,0,1)) / 255. 
        input_disparity = torch.from_numpy(input_disparity.transpose(2,0,1))
        input_boundary = torch.from_numpy(input_boundary.transpose(2,0,1)) / 255.
        input_mask = torch.from_numpy(input_mask.transpose(2,0,1)) / 255.
        gt = torch.from_numpy(gt.transpose(2,0,1)) / 255. 
        return torch.cat([input_x,input_disparity, input_boundary], 0).float(), input_mask.float(), gt.float()


    
    def __getitem__(self, index):
        return self.data_process(self.data_list[index])
        

