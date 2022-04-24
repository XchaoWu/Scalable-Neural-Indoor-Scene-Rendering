import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import os,cv2,sys 
import datetime, time 
import camera
import tools 
import utils 
from tqdm import tqdm 
sys.path.append('../')
from camera import PinholeCamera
from src import dilate_boundary,transparency_statistic 
from sh import eval_sh

class Tile:
    def __init__(self, cfg, tileIdx, tile_center, logdir, device,
                 norm_diffuse_func, norm_specular_func):

        self.logdir = logdir

        self.norm_diffuse_point = norm_diffuse_func
        self.norm_specular_point = norm_specular_func

        self.tileIdx = tileIdx
        
        self.tile_size = cfg.NUM_VOXEL * cfg.VOXEL_SIZE 
        self.voxel_size = cfg.VOXEL_SIZE
        self.num_voxel = cfg.NUM_VOXEL
        self.dilate_num_voxel = self.num_voxel + 2 

        self.tile_center = tile_center
        self.tile_center_np = tile_center

        self.min_corner = self.tile_center - self.tile_size / 2. - self.voxel_size

        self.nsamples = cfg.NSAMPLES 
        self.bgsamples = cfg.BGSAMPLE

        self.device = device

        self.cfg = cfg 

        self.train_ctx = False

        self.mode = 'infer'

        self.cfg.NUM_BASIS = (self.cfg.DEG+1) ** 2
        
        self.voxel_dim = 4 

        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.leaky_relu = torch.nn.LeakyReLU(0.01)

        self.empty = False
    
    def cuda(self):
        func = lambda x: torch.from_numpy(x).to(self.device)
        self.tile_center = func(self.tile_center).float()
        self.min_corner = func(self.min_corner).float()

        try:
            self.voxels = self.voxels.to(self.device)
        except:
            pass 

        try:
            self.nodes_flag = self.nodes_flag.to(self.device)
        except:
            pass 
    
            

    def build_training_context(self, init_nodes_flag):
        self.mode = 'train'

        self.train_error = False 
        self.dilate_num_voxel = self.num_voxel + 2
        self.min_corner = self.tile_center - self.tile_size / 2. - self.voxel_size 

        self.output_dir = os.path.join(self.logdir, f'tile-{self.tileIdx}')
        if os.path.exists(self.output_dir) is False:
            os.makedirs(self.output_dir)


        self.total_step = 0

        self.cuda()

        self.regular_indices = self.create_meshgrid(self.dilate_num_voxel, 
                                                    self.dilate_num_voxel, 
                                                    self.dilate_num_voxel).to(self.device)
        self.regular_sample = (self.regular_indices.float() + 0.5) * self.voxel_size + self.min_corner
        self.zeros = torch.zeros((self.dilate_num_voxel, self.dilate_num_voxel, self.dilate_num_voxel, self.voxel_dim),
                                  dtype=torch.float32, device=self.device)

        if self.cfg.PRETRAINED == '':
            self.build_nodes_flag(self.dilate_num_voxel)
            if self.cfg.INIT_VOXEL:
                self.init_nodes_flag = init_nodes_flag.clone().to(self.device)
                nodes_flag = torch.ones_like(self.nodes_flag) * -1 
                nodes_flag[1:-1,1:-1,1:-1] = self.init_nodes_flag.clone()
                self.nodes_flag = nodes_flag.contiguous()
                dilate_boundary(self.nodes_flag, self.nodes_flag.shape[0])


        if self.train_error:
            return
    
        imgIdxs = np.array(list(np.where(self.cfg.VisImg[self.tileIdx]==1)[0]))

        render_para = self.cfg.TRAIN_RENDER
        face = np.mean(self.cfg.C2Ws[imgIdxs,:,2], axis=0)
        kwargs = {
            "height": render_para.HEIGHT,
            "width": render_para.WIDTH,
            "near": render_para.NEAR,
            "dis": render_para.DIS,
            "up": np.array(render_para.UP),
            "num": render_para.NUM,
            "mode": render_para.MODE,
            "device": self.device,
            "C2Ws": self.cfg.C2Ws[imgIdxs],
            "focus": self.tile_center_np,
            "face": face,
            "zrate": 0.5,
            "rads": np.array(render_para.RADS),
            "scale": render_para.SCALE,
            "focal": self.cfg.Ks[0,0,0],
            "cx": self.cfg.Ks[0,0,2],
            "cy": self.cfg.Ks[0,1,2]}

        self.build_rendering_camera(**kwargs)

        self.train_ctx = True 

    
    def build_nodes_flag(self, length):
        """build nodes flag 
        """
        self.nodes_flag = torch.ones((length,length,length),
                                     dtype=torch.int16,
                                     device=self.device)
    
    def build_rendering_camera(self, **kwargs):
        mode = kwargs['mode']
        distance = kwargs["near"] + self.tile_size * kwargs["dis"]
        kwargs['distance'] = distance

        if mode == '360':
            self.render_cameras = camera.compute_360_camera_path(**kwargs)
        elif mode == 'forward':
            self.render_cameras = camera.compute_spiral_camera_path(**kwargs)
        elif mode == 'llff':
            self.render_cameras = camera.compute_spiral_camera_path_LLFF(**kwargs)
        elif mode == 'inter':
            self.render_cameras = camera.compute_inside_camera_path(**kwargs)
        else:
            raise NotImplementedError

        # self.export_render_camera()
    
    def export_cameras(self, save_path, C2Ws):
        points = tools.cameras_scatter(C2Ws[:,:3,:3].transpose(0,2,1), C2Ws[:,:3,3])
        tools.points2obj(save_path, points)
    
    def export_render_camera(self):
        C2Ws = []
        for cam in self.render_cameras:
            C2Ws.append(cam.C2W.detach().cpu().numpy())
        C2Ws = np.array(C2Ws)
        self.export_cameras("camera_render.obj", C2Ws)
    

    def inference_diffuse(self, model, x):
        
        x = self.norm_diffuse_point(x)

        ori_shape = list(x.shape[:-1])
        # B x (3+cfg) B x 1
        raw = model(x.reshape(-1,3))
        # if torch.sum(torch.isnan(raw)) != 0:
        #     print(torch.sum(torch.isnan(x)))
        #     print(x.max(), x.min())
        # assert(torch.sum(torch.isnan(raw)) == 0)
        rgb_raw = raw[...,:-1]
        sigma_raw = raw[...,-1:]

        rgb_raw = rgb_raw.reshape(*ori_shape,rgb_raw.shape[-1])
        sigma_raw = sigma_raw.reshape(*ori_shape,1)

        rgb = self.sigmoid(rgb_raw)
        sigma = self.relu(sigma_raw)
        return rgb, sigma 
    
    def inference_specular(self, model, x, viewdirs, deg):

        x = self.norm_specular_point(x)

        # print("specular ", x.max(), x.min(), x.mean())

        # input()

        ori_shape = list(x.shape[:-1])
        
        raw = model(x.reshape(-1,3))
        coeffi = raw[...,:-1]
        sigma_raw = raw[...,-1:]

        coeffi = coeffi.reshape(*ori_shape, coeffi.shape[-1])
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        specular = eval_sh(deg, \
                           coeffi.reshape(*coeffi.shape[:-1],self.cfg.NUM_BASIS, -1).transpose(-1,-2),
                           viewdirs[:,None])
        specular = self.sigmoid(specular)
        sigma = self.relu(sigma_raw)

        specular = specular.reshape(*ori_shape, specular.shape[-1])
        sigma = sigma.reshape(*ori_shape, sigma.shape[-1])

        return specular, sigma, coeffi
    
    def query_voxel(self, x):
        """world space x [B x N x 3]
        """
        indices = (x - self.min_corner) / self.voxel_size - 0.5 
        grids = indices / (self.dilate_num_voxel - 1) * 2 - 1
        grids = grids[None, None, ...]

        rgba = F.grid_sample(self.voxels, grids, padding_mode="zeros", align_corners=True)
        rgba = rgba[0,:,0,...].permute(1,2,0)
        rgb = rgba[...,:-1]
        sigma = rgba[...,-1:]
        return rgb, sigma 

    @staticmethod
    def create_meshgrid(D,H,W):
        Zs,Ys,Xs = torch.meshgrid(torch.arange(D), torch.arange(H),  torch.arange(W))
        return torch.stack([Xs,Ys,Zs], dim=-1)
    

    def compute_voxels_v2(self, model, density=4):
        regular_indices = self.create_meshgrid(self.dilate_num_voxel * density,
                                               self.dilate_num_voxel * density,
                                               self.dilate_num_voxel * density).to(self.device)
        offset_center = 1. / density / 2. 
        grid_size = self.voxel_size / density
        # 4N x 4N x 4N x 3
        regular_sample = (regular_indices.float() + offset_center) * grid_size + self.min_corner

        nodes_flag = self.nodes_flag.clone()
        nodes_flag = nodes_flag.repeat_interleave(density, dim=0)
        nodes_flag = nodes_flag.repeat_interleave(density, dim=1)
        nodes_flag = nodes_flag.repeat_interleave(density, dim=2)
        non_empty = torch.where(nodes_flag != -1)
        regular_sample = regular_sample[non_empty].reshape(-1,3)

        batchsize = 16777216
    
        rgb_list = []
        sigma_list = []
        for i in range(0, regular_sample.shape[0], batchsize):
            temp_rgb, temp_sigma = self.inference_diffuse(model, regular_sample[i:i+batchsize])
            rgb_list.append(temp_rgb)
            sigma_list.append(temp_sigma)

        regular_rgb = torch.cat(rgb_list, 0)
        regular_sigma = torch.cat(sigma_list, 0)
        # regular_rgb, regular_sigma = self.inference_diffuse(model, regular_sample)

        voxels = torch.zeros((self.dilate_num_voxel * density, 
                              self.dilate_num_voxel * density, 
                              self.dilate_num_voxel * density, 
                              self.voxel_dim),
                             dtype=torch.float32, device=self.device)
        voxels[non_empty] = torch.cat([regular_rgb, regular_sigma], dim=-1)
        voxels = voxels[None, ...].permute(0,4,1,2,3) # 1 x 4 x D x H x W
        avg_pool = nn.AvgPool3d(kernel_size=density, stride=density, padding=0)
        self.voxels = avg_pool(voxels)

    def compute_voxels(self, model, noise=False):
        # N x N x N x 3 
        regular_sample = self.regular_sample.clone()
        voxels = self.zeros.clone()
        non_empty = torch.where(self.nodes_flag != -1)
        regular_sample = regular_sample[non_empty]
        if noise:
            regular_sample = regular_sample + torch.rand_like(regular_sample) * 0.1 * self.voxel_size
        regular_rgb, regular_sigma = self.inference_diffuse(model, regular_sample.reshape(-1,3))
        voxels[non_empty] = torch.cat([regular_rgb, regular_sigma], dim=-1)
        assert(torch.sum(torch.isnan(regular_rgb)) == 0)
        assert(torch.sum(torch.isnan(regular_sigma)) == 0)
        self.voxels = voxels[None,...].permute(0,4,1,2,3) # 1 x 4 x D x H x W
    
    @torch.no_grad()
    def pruning(self, model, data, block=4, th=0.5, T_th = 0, rate=0.1):

        T_voxels = torch.zeros(self.dilate_num_voxel, self.dilate_num_voxel, self.dilate_num_voxel, 
                               dtype=torch.float32, device=self.device)
        sample_step = self.voxel_size * rate 

        sigma_voxels = self.relu(self.voxels[0,3].permute(2,1,0).contiguous())
        
        batchSize = self.cfg.PATCH_SIZE ** 2
        for i in range(0, data.shape[0], batchSize):
            batch_data = data[i:i+batchSize]
            # B x 3 
            rays_o = batch_data[...,:3].reshape(-1,3).contiguous()
            rays_d = batch_data[...,3:6].reshape(-1,3).contiguous()
            if self.cfg.PATCH_TRAIN:
                # PATCHSIZE**2 x 1
                mask = batch_data[...,13:14].reshape(-1,1).contiguous() 
                valid = torch.where(mask == 1)[0]
                rays_o = rays_o[valid]
                rays_d = rays_d[valid]

            transparency_statistic(rays_o, rays_d, self.dilate_num_voxel, self.tile_center,
                                  self.tile_size, self.voxel_size, sample_step, 
                                  sigma_voxels, self.nodes_flag.permute(2,1,0).contiguous(), 
                                  T_voxels)
        T_voxels = T_voxels.permute(2,1,0)


        min_T = torch.min(T_voxels).cpu().numpy()
        max_T = torch.max(T_voxels).cpu().numpy()
        mean_T = torch.mean(T_voxels).cpu().numpy()
        info = f"Tile {self.tileIdx} Transparency info:\n"
        info += f"block size {block}\tgeo_th: {th}\ttrans_th: {T_th}\n"
        info += f"min {min_T:.5f}\tmax {max_T:.5f}\tmean {mean_T:.5f}\n"
        
        # tools.output_T_voxel('T.obj', T_voxels.detach().cpu().numpy().transpose(2,1,0), 
        #                     self.tile_center_np, self.tile_size, self.voxel_size, T_th)    
        T_voxels = torch.nn.MaxPool3d(kernel_size=3, stride=1, padding=1)(T_voxels[None,None])
        T_voxels = T_voxels[0,0]

        block_length = self.num_voxel // block 
        regular_indices = self.create_meshgrid(block_length, block_length, block_length).to(self.device)
        # D x H x W x 3
        regular_o = (regular_indices.float() * block + 0.5) * self.voxel_size + self.min_corner + self.voxel_size
        # DHW x 3
        regular_o = regular_o.reshape(-1,3)

        # total_samples = []

        # GGG x 3 
        offsets = self.create_meshgrid(block, block, block).reshape(-1,3).to(self.device)

        # GGG x DHW x 3 
        regular_sample = regular_o[None,:] + offsets[:,None] * self.voxel_size

        _, regular_sigma = self.inference_diffuse(model, regular_sample)

        regular_sigma = regular_sigma.reshape(block**3, block_length, block_length, block_length)

        regular_sigma, _ = torch.max(regular_sigma, dim=0)
        regular_sigma = regular_sigma.repeat_interleave(block,dim=0)
        regular_sigma = regular_sigma.repeat_interleave(block,dim=1)
        regular_sigma = regular_sigma.repeat_interleave(block,dim=2)

        # nodes_flag = self.nodes_flag.clone()
        # nodes_flag = torch.ones_like(self.nodes_flag)
        if self.cfg.INIT_VOXEL:
            nodes_flag = torch.ones_like(self.nodes_flag) * -1 
            nodes_flag[1:-1,1:-1,1:-1] = self.init_nodes_flag.clone()
            nodes_flag = nodes_flag.contiguous()
        else:
            nodes_flag = torch.ones_like(self.nodes_flag) * -1 
            nodes_flag[1:-1,1:-1,1:-1] = 1

        non_overlap = nodes_flag[1:-1,1:-1,1:-1]
        non_overlap[torch.where(torch.exp(-regular_sigma) > th)] = -1

        non_overlap[torch.where(T_voxels[1:-1,1:-1,1:-1] <= T_th)] = -1

        dilate_boundary(nodes_flag, nodes_flag.shape[0])

        self.nodes_flag = nodes_flag

        ori_voxel_num = self.num_voxel**3 
        real_voxel_num = len(torch.where(non_overlap == 1)[0])

        info += f'\noriginal voxel num: {ori_voxel_num}\nafter pruning: {real_voxel_num}\n'
        geo_rate = real_voxel_num / ori_voxel_num * 100
        info += f'Geometry Rate: {geo_rate:.3f} %\n'

        print(info)

        if real_voxel_num == 0:
            self.train_error = True
            self.error_mode = "Empty Geometry After Pruning Error"
            with open(os.path.join(self.logdir, 'errorTiles.txt'), 'a') as f:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
                f.write(f"Time {current_time}\ttileIdx: {self.tileIdx}\tError: {self.error_mode}\n")

    
    def load_voxel(self, path):
        voxels = np.load(path)
        voxels = torch.from_numpy(voxels)
        # 1 x C x D x H x W 
        self.voxels = voxels[None,...].permute(0,4,3,2,1)
        self.voxels = self.voxels.to(self.device)
    
    def load_nodes_flag(self, path):
        nodes_flag = np.load(path)
        nodes_flag = torch.from_numpy(nodes_flag)
        self.nodes_flag = torch.zeros_like(nodes_flag)
        self.nodes_flag[1:-1,1:-1,1:-1] = nodes_flag[1:-1,1:-1,1:-1].permute(2,1,0)
        self.nodes_flag = self.nodes_flag.contiguous()
        self.nodes_flag = self.nodes_flag.to(self.device)
        self.init_nodes_flag = self.nodes_flag[1:-1,1:-1,1:-1].clone().contiguous()
        # self.nodes_flag = nodes_flag.permute(2,1,0)

    def save(self, epoch):

        voxel_save_path = os.path.join(self.output_dir,f"Voxel-T{self.tileIdx}-E{epoch}.npy")
        node_save_path = os.path.join(self.output_dir, f"Node-T{self.tileIdx}-E{epoch}.npy")

        try:
            np.save(voxel_save_path, self.voxels.detach().cpu().numpy().squeeze().transpose(3,2,1,0))
        except:
            print("Voxels saved failed!")
        else:
            print(f"Voxels saved successfully to {voxel_save_path}")

        try:
            np.save(node_save_path, self.nodes_flag.detach().cpu().numpy().transpose(2,1,0))
        except:
            print("Nodes saved failed!")
        else:
            print(f"Nodes saved successfully to {node_save_path}")


    def infer_video(self, model, epoch, fps=15, save_img=True, save_path=None):
        preds = []
        print(f'Rendering tile {self.tileIdx} to video ...\n')
        for cam in self.render_cameras:
            pred = self.render_tile(model, cam, epoch)
            preds.append(pred)
        if self.mode == 'train':
            tools.generate_video(os.path.join(self.output_dir,f"{self.tileIdx}-{epoch}.mp4"), preds, fps=fps)
            if save_img:
                tools.save_img_list(self.output_dir, preds)
        elif self.mode == 'infer':
            tools.generate_video(os.path.join(save_path,f"{self.tileIdx}-{epoch}.mp4"), preds, fps=fps)
            if save_img:
                tools.save_img_list(save_path, preds)

    def infer_gif(self, model, epoch, fps=15, save_img=True, save_path=None):
        preds = []
        print(f'Rendering tile {self.tileIdx} to GIF ...\n')
        for cam in tqdm(self.render_cameras):
            pred = self.render_tile(model, cam, epoch)
            preds.append(pred)
        
        if self.mode == 'train':
            tools.generate_gif(os.path.join(self.output_dir,f"{self.tileIdx}-{epoch}.gif"), preds, fps=fps)
            if save_img:
                tools.save_img_list(self.output_dir, preds)
        elif self.mode == 'infer':
            tools.generate_gif(os.path.join(save_path,f"{self.tileIdx}.gif"), preds, fps=fps)
            if save_img:
                tools.save_img_list(save_path, preds)

    @torch.no_grad()
    def render_tile(self, model, cam:PinholeCamera, epoch):
        rays_o, rays_d = cam.get_rays()

        z_vals, sparse_dists = utils.sparse_sampling(rays_o, rays_d, self.nodes_flag[1:-1,1:-1,1:-1],
                                                     self.num_voxel, self.tile_center_np, self.tile_size, 
                                                     self.nsamples)
        

        # invalid = torch.any(z_vals == -1, dim=-1)

        samples = rays_o[:,None,:] + z_vals[...,None] * rays_d[:,None,:]


        dists = z_vals[...,1:] - z_vals[...,:-1]

        dists = sparse_dists[:,:-1]
        
        # dists[invalid] = 0

        rgb, sigma = self.query_voxel(samples)

        diffuse, depth_map, _, _ = utils.volume_rendering_v2(rgb[:,:-1,:], sigma[:,:-1,:], z_vals[:,:-1,None], dists[...,None])
        # diffuse[invalid] = 0
        # reflection_z_vals = utils.inverse_z_sample(depth_map, self.cfg.far, self.cfg.RSAMPLES)
        # trans_decay = (1 + np.exp(-0.1 * epoch)) / 2. 
        reflection_z_vals = utils.reflection_sampling(rays_o, rays_d, self.nodes_flag[1:-1,1:-1,1:-1].clone(), self.voxels[0].clone(), self.num_voxel,
                                                    self.tile_center_np, self.tile_size, self.cfg.RSAMPLES, self.cfg.far, 1.0)
        # reflection_z_vals = utils.inverse_z_sample(z_vals[..., -1:]+self.cfg.RSAMPLES_OFFSET*self.tile_size, self.cfg.far, self.cfg.RSAMPLES)
        # print(reflection_z_vals.shape)
        # [BUG FIX ME]
        invalid = torch.any(reflection_z_vals == -1, dim=-1) 
        # print(torch.sum(invalid))
        reflection_samples = rays_o[:,None,:] + reflection_z_vals[...,None] * rays_d[:,None,:]
        reflection, reflection_sigma, _ = self.inference_specular(model, reflection_samples, rays_d, self.cfg.DEG)
        reflection_dists = torch.cat([reflection_z_vals[...,1:] - reflection_z_vals[...,:-1], 
                                        1e10*torch.ones(rays_o.shape[0],1, device=self.device)], dim=-1)
        rgb_vd,_,_ = utils.volume_rendering(reflection, reflection_sigma, reflection_dists[...,None])

        rgb_vd[invalid] = 0
        output = torch.clamp(diffuse+rgb_vd, 0, 1)

        diffuse = diffuse.reshape(cam.height, cam.width, 3).cpu().detach().numpy().clip(0,1) * 255
        rgb_vd = rgb_vd.reshape(cam.height, cam.width, 3).cpu().detach().numpy().clip(0,1) * 255
        output = output.reshape(cam.height, cam.width, 3).cpu().detach().numpy().clip(0,1) * 255
        depth_map = depth_map.reshape(cam.height, cam.width, 1).cpu().repeat(1,1,3).numpy()
        depth_map = depth_map / (depth_map.max() + 0.00001) * 255 

        return np.concatenate([output, diffuse, rgb_vd, depth_map], axis=1)
    

    def train_batch_sec(self, epoch, batch_data, log_img, **kwargs):
        if self.train_error:
            print(f"\n====== Current Tile {self.tileIdx} has train error, return ======\n")
            return dict(), False
        
        criterions = kwargs['criterions']

        optimizer = kwargs["optimizer"]

        coeffi_model = kwargs["coeffi"]

        forward_time = 0
        backward_time = 0

        start_time = time.time()

        batchSize = batch_data.shape[0]
        rays_o = batch_data[...,:3].reshape(-1,3).contiguous()
        rays_d = batch_data[...,3:6].reshape(-1,3).contiguous()
        near = batch_data[...,6:7].reshape(-1,1).contiguous() # B x image_size x 1 
        gtcolor = batch_data[...,7:10].reshape(-1,3).contiguous()  
        gt_diffuse = batch_data[...,10:13].contiguous()  
        mask = batch_data[...,13:14].reshape(-1,1).contiguous() 

        # loss_weight = batch_data[...,14:15].reshape(-1,1).contiguous() 
        # mirror_depth = batch_data[...,14:15].reshape(-1,1).contiguous() 

        # invalid = torch.any(mirror_depth == -1, dim=-1)
        # mask[invalid] = 0
        invalid = torch.where(mask==0)[0]
        valid  = torch.where(mask==1)[0]

        # B x N
        reflection_z_vals = utils.inverse_z_sample(near, self.cfg.far, self.cfg.RSAMPLES)

        # if self.cfg.RIMPORTANCE > 0:
        #     importance_z_vals = utils.important_reflection_sample(mirror_depth, self.cfg.RIMPORTANCE, epsilon=self.tile_size*0.5)
        #     reflection_z_vals, _  = torch.sort(torch.cat([reflection_z_vals, importance_z_vals], -1), -1)

        reflection_samples = rays_o[:,None,:] + reflection_z_vals[...,None] * rays_d[:,None,:]

        ts = time.time()
        
        if epoch > self.cfg.KN_EPOCH:
            reflection, reflection_sigma, coeffi = self.inference_specular(coeffi_model, reflection_samples, rays_d, self.cfg.DEG)
        else:
            reflection, reflection_sigma, coeffi = self.inference_specular(coeffi_model, reflection_samples, rays_d, 0)

        torch.cuda.synchronize()
        te = time.time()
        forward_time += (te-ts)

        coeffi = coeffi.reshape(*coeffi.shape[:-1],self.cfg.NUM_BASIS, -1).transpose(-1,-2) # B x NSAMPLES x 3 x NUM_BASIS  
        reflection_dists = torch.cat([reflection_z_vals[...,1:] - reflection_z_vals[...,:-1], 
                                        1e10*torch.ones(rays_o.shape[0],1, device=self.device)], dim=-1)
        reflection_dists[invalid] = 0
        rgb_vd,_,_ = utils.volume_rendering(reflection, reflection_sigma, reflection_dists[...,None])
        rgb_vd = rgb_vd.reshape(batchSize, self.cfg.PATCH_SIZE,self.cfg.PATCH_SIZE,3)
        mask = mask.reshape(batchSize, self.cfg.PATCH_SIZE,self.cfg.PATCH_SIZE,1)
        gtcolor = gtcolor.reshape(batchSize, self.cfg.PATCH_SIZE,self.cfg.PATCH_SIZE,3)
        gt_diffuse = gt_diffuse.reshape(batchSize, self.cfg.PATCH_SIZE,self.cfg.PATCH_SIZE,3)

        pred = torch.clamp(gt_diffuse + rgb_vd, 0, 1)


        # temp = reflection_samples[valid]
        # temp = temp.reshape(-1,3).detach().cpu().numpy()
        # colors = np.ones_like(temp) * (255,0,0)
        # pts = np.concatenate([temp, colors], axis=-1)
        # tools.points2obj("points.obj", pts)
        # temp = gtcolor[0].clone()
        # temp = temp.detach().cpu().numpy()
        # temp_mask = mask[0].detach().cpu().numpy()
        # temp = temp * temp_mask
        # cv2.imwrite("hh.png", temp[...,::-1] * 255)
        # input()

        out_info = {}

        loss = criterions.compute_loss(pred=pred, gtcolor=gtcolor, coeffi=coeffi, 
                                       mask=mask, epoch=epoch)
        ts = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        te = time.time()
        backward_time += (te-ts)  

        end_time = time.time()

        out_info['forward_time'] = forward_time
        out_info['backward_time'] = backward_time
        out_info['batch_time'] = end_time - start_time 


        if log_img:
            temp_mask = mask[0].detach().cpu().numpy()
            temp_pred = pred[0].detach().cpu().numpy() 
            temp_gtcolor = gtcolor[0].detach().cpu().numpy()
            temp_gtdiffuse = gt_diffuse[0].detach().cpu().numpy()
            
            # print(temp_diffuse.shape, temp_gt_diffuse.shape)
            temp_out = np.concatenate([temp_gtdiffuse, temp_pred, temp_gtcolor], axis=1)
            temp_out2 = np.concatenate([temp_gtdiffuse*temp_mask,
                                        temp_pred*temp_mask, 
                                        temp_gtcolor*temp_mask], axis=1)
            out = np.concatenate([temp_out, temp_out2], axis=0)
            formal_time = datetime.datetime.now().strftime("%d-%H-%M")
            save_path = os.path.join(self.output_dir, f"train-E{epoch}-{formal_time}.png")
            cv2.imwrite(save_path, out[...,::-1] * 255)
        
        return out_info, True

    def train_batch(self, epoch, batch_data, log_img, train_FLAG, nsamples, **kwargs):

        if self.train_error:
            # print(f"\n====== Current Tile {self.tileIdx} has train error, return ======\n")
            return dict(), False
        
        self.nsamples = nsamples

        """
        train_FLAG 0  only diffuse 
        train_FLAG 1  only specular 
        train_FLAG 2  both
        """

        assert train_FLAG in [0,1,2]

        criterions = kwargs['criterions']

        optimizer = kwargs["optimizer"]
        model = kwargs["model"]
        coeffi_model = kwargs["coeffi"]


        forward_time = 0
        backward_time = 0

        start_time = time.time()

        batchSize = batch_data.shape[0]
        rays_o = batch_data[...,:3].reshape(-1,3).contiguous()
        rays_d = batch_data[...,3:6].reshape(-1,3).contiguous()
        bgdepth = batch_data[...,6:7].reshape(-1,1).contiguous() # B x image_size x 1 
        gtcolor = batch_data[...,7:10].reshape(-1,3).contiguous()  
        gt_diffuse = batch_data[...,10:13].contiguous()  
        mask = batch_data[...,13:14].reshape(-1,1).contiguous() 

        z_vals, sparse_dists = utils.sparse_sampling(rays_o, rays_d, self.nodes_flag[1:-1,1:-1,1:-1],
                                                        self.num_voxel, self.tile_center_np, self.tile_size, 
                                                        self.nsamples)

        invalid = torch.any(z_vals == -1, dim=-1)
        mask[invalid] = 0 # 不参与 backward 
        if (torch.sum(mask) == 0):
            return dict(), False
        invalid = torch.where(mask==0)[0]
        valid = torch.where(mask==1)[0]

        mask = mask.reshape(batchSize, self.cfg.PATCH_SIZE,self.cfg.PATCH_SIZE,1)
        gtcolor = gtcolor.reshape(batchSize, self.cfg.PATCH_SIZE,self.cfg.PATCH_SIZE,3)
        gt_diffuse = gt_diffuse.reshape(batchSize, self.cfg.PATCH_SIZE,self.cfg.PATCH_SIZE,3)

        if train_FLAG == 0 or train_FLAG == 2:
            bg_zvals = utils.sampling_background(rays_o, rays_d, bgdepth, self.tile_center, self.tile_size, self.bgsamples, 
                                                    self.cfg.BGSAMPLE_RANGE)
            bg_sample = rays_o[:,None,:] + bg_zvals[...,None] * rays_d[:,None,:]

            ts = time.time()
            # B x NB x 3 B x NB x 1
            bg_rgb, bg_sigma = self.inference_diffuse(model, bg_sample)
            self.compute_voxels(model)
            assert(torch.sum(torch.isnan(self.voxels)) == 0)
            torch.cuda.synchronize()
            te = time.time()
            forward_time += (te-ts)
            samples = rays_o[:,None,:] + z_vals[...,None] * rays_d[:,None,:]
            rgb, sigma = self.query_voxel(samples)
            total_z_vals = torch.cat([z_vals, bg_zvals], dim=-1)
            dists = torch.cat([total_z_vals[...,1:] - total_z_vals[...,:-1], 1e10*torch.ones(rays_o.shape[0],1, device=self.device)], dim=-1)
            dists[:,:self.nsamples] = sparse_dists
            dists[invalid] = 0
            assert(torch.min(dists) >= 0)
            
            diffuse_intile, left_trans, _ = utils.volume_rendering(rgb, sigma, dists[:,:self.nsamples, None])
            diffuse_bg, _, _ = utils.volume_rendering(bg_rgb, bg_sigma, dists[:,self.nsamples:, None], left_trans[:,-1:,:])
            diffuse_intile = diffuse_intile.reshape(batchSize, self.cfg.PATCH_SIZE,self.cfg.PATCH_SIZE,3)
            diffuse_bg = diffuse_bg.reshape(batchSize, self.cfg.PATCH_SIZE,self.cfg.PATCH_SIZE,3)
            bgdepth = bgdepth.reshape(batchSize, self.cfg.PATCH_SIZE,self.cfg.PATCH_SIZE,1)
            bg_mask = bgdepth != -1
            diffuse = diffuse_intile + diffuse_bg * bg_mask # [FIX THIS]
            # pred = diffuse
        else:
            with torch.no_grad():
                bg_zvals = utils.sampling_background(rays_o, rays_d, bgdepth, self.tile_center, self.tile_size, self.bgsamples, 
                                                        self.cfg.BGSAMPLE_RANGE)
                bg_sample = rays_o[:,None,:] + bg_zvals[...,None] * rays_d[:,None,:]

                ts = time.time()
                # B x NB x 3 B x NB x 1
                bg_rgb, bg_sigma = self.inference_diffuse(model, bg_sample)
                self.compute_voxels(model)
                assert(torch.sum(torch.isnan(self.voxels)) == 0)
                torch.cuda.synchronize()
                te = time.time()
                forward_time += (te-ts)
                samples = rays_o[:,None,:] + z_vals[...,None] * rays_d[:,None,:]
                rgb, sigma = self.query_voxel(samples)
                total_z_vals = torch.cat([z_vals, bg_zvals], dim=-1)
                dists = torch.cat([total_z_vals[...,1:] - total_z_vals[...,:-1], 1e10*torch.ones(rays_o.shape[0],1, device=self.device)], dim=-1)
                dists[:,:self.nsamples] = sparse_dists
                dists[invalid] = 0
                assert(torch.min(dists) >= 0)
                
                diffuse_intile, left_trans, _ = utils.volume_rendering(rgb, sigma, dists[:,:self.nsamples, None])
                diffuse_bg, _, _ = utils.volume_rendering(bg_rgb, bg_sigma, dists[:,self.nsamples:, None], left_trans[:,-1:,:])
                diffuse_intile = diffuse_intile.reshape(batchSize, self.cfg.PATCH_SIZE,self.cfg.PATCH_SIZE,3)
                diffuse_bg = diffuse_bg.reshape(batchSize, self.cfg.PATCH_SIZE,self.cfg.PATCH_SIZE,3)
                bgdepth = bgdepth.reshape(batchSize, self.cfg.PATCH_SIZE,self.cfg.PATCH_SIZE,1)
                bg_mask = bgdepth != -1
                diffuse = diffuse_intile + diffuse_bg * bg_mask # [FIX THIS]
        
        if train_FLAG == 1 or train_FLAG == 2:
            # trans_decay = (1 + np.exp(-0.1 * epoch)) / 2. 
            reflection_z_vals = utils.reflection_sampling(rays_o, rays_d, self.nodes_flag[1:-1,1:-1,1:-1].clone(), self.voxels[0].clone(),self.num_voxel,
                                                        self.tile_center_np, self.tile_size, self.cfg.RSAMPLES, self.cfg.far, 1.0)
            # reflection_z_vals = utils.inverse_z_sample(z_vals[..., -1:]+self.cfg.RSAMPLES_OFFSET*self.tile_size, self.cfg.far, self.cfg.RSAMPLES)
            reflection_mask = torch.all(reflection_z_vals != -1, dim=-1)

            reflection_samples = rays_o[:,None,:] + reflection_z_vals[...,None] * rays_d[:,None,:]
            ts = time.time()
            reflection, reflection_sigma, coeffi = self.inference_specular(coeffi_model, reflection_samples, rays_d, self.cfg.DEG)
            coeffi = coeffi.reshape(*coeffi.shape[:-1],self.cfg.NUM_BASIS, -1).transpose(-1,-2) # B x NSAMPLES x 3 x NUM_BASIS  
            torch.cuda.synchronize()
            te = time.time()
            forward_time += (te-ts)

            reflection_dists = torch.cat([reflection_z_vals[...,1:] - reflection_z_vals[...,:-1], 
                                            1e10*torch.ones(rays_o.shape[0],1, device=self.device)], dim=-1)
            reflection_dists[invalid] = 0
            rgb_vd,_,_ = utils.volume_rendering(reflection, reflection_sigma, reflection_dists[...,None])
            rgb_vd = rgb_vd.reshape(batchSize, self.cfg.PATCH_SIZE,self.cfg.PATCH_SIZE,3)
            reflection_mask = reflection_mask.reshape(batchSize, self.cfg.PATCH_SIZE,self.cfg.PATCH_SIZE,1)

        if train_FLAG == 1:
            pred = torch.clamp(diffuse.detach() + reflection_mask * rgb_vd, 0, 1)
        elif train_FLAG == 2:
            pred = torch.clamp(diffuse + reflection_mask * rgb_vd, 0, 1) 
        else:
            pred = diffuse
            coeffi = None 
    

        out_info = {}

        loss = criterions.compute_loss(pred=pred, gtcolor=gtcolor, coeffi=coeffi, 
                                       diffuse_intile=diffuse_intile,
                                       gt_diffuse=gt_diffuse,
                                       mask=mask, epoch=epoch)
        # loss = loss + 1.0 * torch.mean(1-torch.exp(-reflection_sigma[intile_reflection_loc]))

        ts = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        te = time.time()
        backward_time += (te-ts)  

        end_time = time.time()

        out_info['forward_time'] = forward_time
        out_info['backward_time'] = backward_time
        out_info['batch_time'] = end_time - start_time 

        if log_img:
            temp_diffuse = diffuse[0].detach().cpu().numpy()
            temp_mask = mask[0].detach().cpu().numpy()
            temp_bg_mask = bg_mask[0].detach().repeat(1,1,3).cpu().numpy()
            temp_pred = pred[0].detach().cpu().numpy() 
            temp_gtcolor = gtcolor[0].detach().cpu().numpy()
            temp_gtdiffuse = gt_diffuse[0].detach().cpu().numpy()
            temp_bgdiffuse = diffuse_bg[0].detach().cpu().numpy()
            temp_forediffuse = diffuse_intile[0].detach().cpu().numpy()
            
            # print(temp_diffuse.shape, temp_gt_diffuse.shape)
            temp_out = np.concatenate([temp_bgdiffuse, temp_bg_mask, temp_forediffuse, temp_gtdiffuse, temp_diffuse, temp_pred, temp_gtcolor], axis=1)
            temp_out2 = np.concatenate([temp_bgdiffuse*temp_mask,
                                        temp_bg_mask,
                                        temp_forediffuse*temp_mask,
                                        temp_gtdiffuse*temp_mask,
                                        temp_diffuse*temp_mask, 
                                        temp_pred*temp_mask, 
                                        temp_gtcolor*temp_mask], axis=1)
            out = np.concatenate([temp_out, temp_out2], axis=0)
            formal_time = datetime.datetime.now().strftime("%d-%H-%M")
            save_path = os.path.join(self.output_dir, f"train-E{epoch}-{formal_time}.png")
            cv2.imwrite(save_path, out[...,::-1] * 255)
        
        return out_info, True

