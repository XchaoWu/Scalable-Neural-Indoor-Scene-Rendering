import torch 
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F 
from . import FASTRENDERING
import cv2,os,sys,time,math,pynvml
import networks

class FRender:
    def __init__(self,
                 scene_path, 
                 cnn_path, 
                 height=720, 
                 width=1280,
                 focal=1000,
                 max_tracing_tile=30,
                 reflection_scale=2,
                 origin=[0,8,0],
                 tensorRT=True):

        self.device = torch.device("cuda:0")

        self.Unet_path = cnn_path
        self.reflection_scale = reflection_scale
        self.TRT = tensorRT
        self.focal = focal
        self.max_tracing_tile = max_tracing_tile

        if self.TRT:
            from torch2trt import torch2trt

        pynvml.nvmlInit()

        self.num_thread = 256 
        self.max_pixels_per_block = 64 
        self.ksize_boundary = 3
        self.trans_decay = 0.8
        self.enable_Unet = False 
        self.early_stop_R = 0.01
        self.early_stop_D = 0.01
        self.sample_step_scale = 0.2 
        self.move_scale = 0.08

        self.H = height // 4 * 4
        self.W = width // 4 * 4 
        self.WINDOW_H = self.H  
        self.WINDOW_W = self.W 
        
        self.render_prepare(scene_path)

        self.intrinsic = torch.tensor([self.W / 2., self.H / 2., 1./self.focal], 
                                        device=self.device, dtype=torch.float32)

        self.origin = torch.tensor(origin, dtype=torch.float32, device=self.device)

        self.azimuth = 15
        self.radius = 4
        self.center = [0,0,0]
        self.inclination = 1.5
        self.compute_c2w()


        self.DH = self.H // self.reflection_scale
        self.DW = self.W // self.reflection_scale
        self.downscale_numPixel = self.DH * self.DW
        self.inverse_far = 1.0 / self.far



        self.up = torch.tensor([0,1,0], device=self.device, dtype=torch.float32)
        
        self.frame = torch.empty([self.H, self.W, 3], dtype=torch.float, device=self.device)
        self.frame_pointer = self.frame.data_ptr()

        self.render_mode = 1


        self.enable_Unet = False
        if self.Unet_path != None:
            if os.path.isfile(self.Unet_path):
                self.Unet = networks.CNN().to(self.device)
                self.Unet.load_state_dict(torch.load(self.Unet_path))
                print("\n===== Finishde Loading Unet =====\n")
                self.enable_Unet = True
                if self.TRT:
                    input=torch.randn(1, 5, self.H, self.W, requires_grad=True).to(self.device)
                    self.model_trt = torch2trt(self.Unet.eval(), [input], max_batch_size=1, fp16_mode=True)
                    self.CNN = self.model_trt
                else:
                    self.CNN = self.Unet



    def get_gpuUsed(self):
        info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        print(f'GPU mem used     : {info.used / (1000**3)} GB')

    def render_prepare(self, path, **kwargs):
        render_data = np.load(path, allow_pickle=True)

        mem = 0
        for key in render_data:
            tem_mem = render_data[key].nbytes / (1000 ** 3) 
            print(f"render data {key}\tshape {render_data[key].shape} mem: {tem_mem*1000:.6f} MB")
            mem += tem_mem 
        print(f"\ntotal render data mem {mem:.3f} GB")

        self.octree_depth = render_data['octree_depth'][0]
        self.num_voxel = render_data['num_voxel'][0]
        self.voxel_size = render_data['voxel_size'][0]
        self.scene_min_corner =  torch.from_numpy(render_data['scene_min_corner']).float().to(self.device).contiguous()
        self.tile_shape = torch.from_numpy(render_data['tile_shape']).int().to(self.device).contiguous()
        self.net_params = torch.from_numpy(render_data['net_params']).half().to(self.device).contiguous()
        self.group_centers = torch.from_numpy(render_data['group_centers']).float().to(self.device).contiguous()
        self.centers = torch.from_numpy(render_data['centers']).float().to(self.device).contiguous()
        self.tile2group = torch.from_numpy(render_data['tile2group']).int().to(self.device).contiguous()
        self.tile_IndexMap = torch.from_numpy(render_data['tile_IndexMap']).int().to(self.device).contiguous()
        self.block_IndexMap = torch.from_numpy(render_data['block_IndexMap']).int().to(self.device).contiguous()
        self.nodes_IndexMap = torch.from_numpy(render_data['nodes_IndexMap']).short().to(self.device).contiguous()
        self.nodes_sampleFlag = torch.from_numpy(render_data['nodes_sampleFlag']).bool().to(self.device).contiguous()
        self.voxels_start = torch.from_numpy(render_data['voxels_start']).long().to(self.device).contiguous()
        self.data_voxels = torch.from_numpy(render_data['data_voxels']).half().to(self.device).contiguous()
        self.group_names = render_data['group_names']

        self.get_gpuUsed()

        self.tile_size = self.num_voxel * self.voxel_size
        self.block_num_voxel = self.num_voxel // (2 ** self.octree_depth) + 2

        self.sample_step = self.sample_step_scale * self.voxel_size
        self.max_depth = np.max(render_data['tile_shape'] * self.tile_size)
        self.far = np.linalg.norm(self.tile_size * render_data['tile_shape'])
        self.sample_far = self.far
        self.scene_size = self.tile_shape * self.tile_size
        self.num_group = self.group_centers.shape[0]


    def enable_lr_reflection(self, flag):
        if flag:
            self.reflection_scale = 2
        else:
            self.reflection_scale = 1
        self.DH = self.H // self.reflection_scale
        self.DW = self.W // self.reflection_scale
        self.downscale_numPixel = self.DH * self.DW
        
    def set_early_stop_R(self, early_stop):
        self.early_stop_R = early_stop

    def set_move_scale(self, scale):
        self.move_scale = scale 

    def set_early_stop_D(self, early_stop):
        self.early_stop_D = early_stop

    def set_far(self, scale):
        self.sample_far = self.far * scale 

    def set_max_tracing_tile(self, num):
        self.max_tracing_tile = num
        
    def move_forward(self):
        self.origin = self.origin - torch.cross(self.c2w[:,0], self.up) * self.move_scale
    def move_back(self):
        self.origin = self.origin + torch.cross(self.c2w[:,0], self.up) * self.move_scale
    def move_left(self):
        self.origin = self.origin - self.c2w[:,0] * self.move_scale
    def move_right(self):
        self.origin = self.origin + self.c2w[:,0] * self.move_scale
    def move_up(self):
        self.origin[1] += self.move_scale
    def move_down(self):
        self.origin[1] -= self.move_scale
    def zoom(self, scale):
        self.intrinsic[2] *= scale

    def set_focal_scale(self, scale):
        self.intrinsic[2] = 1. / (self.focal* scale)
    
    def set_sample_scale(self, scale):
        self.sample_step_scale = scale
        self.sample_step = self.sample_step_scale * self.voxel_size
    
    def set_trans_decay(self, v):
        self.trans_decay = v 
    
    def set_num_thread(self, num):
        self.num_thread = num

    def display_groupIdx(self, x, y):
        groupIdx = int(self.large_netIdx[int(y+0.5),int(x+0.5)])
        if groupIdx == -1:
            print("Not group")
            return
        name_of_group = self.group_names[groupIdx]

        print(f"\ngroup name: {name_of_group}\n")

    def set_mode(self, mode_type):
        self.render_mode = mode_type
    def set_unet(self, flag):
        self.enable_Unet = flag

    def rotate(self, delta_x, delta_y):
        self.azimuth += (delta_x / 2)
        self.inclination += (delta_y / 2)
        eps = 0.001
        self.inclination = min(max(eps, self.inclination), math.pi - eps)
        self.compute_c2w()

    def lookat(self, look_from, look_to, tmp = np.asarray([0., -1., 0.])):
        forward = look_from - look_to
        forward = forward / np.linalg.norm(forward)
        right = np.cross(tmp, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        
        c2w_T = np.zeros((3,3))
        c2w_T[0,0:3] = right
        c2w_T[1,0:3] = up
        c2w_T[2,0:3] = forward
        return c2w_T.T

    def compute_c2w(self):
        offset = np.asarray([self.radius * math.cos(-self.azimuth) * math.sin(self.inclination),
                             self.radius * math.cos(self.inclination),
                             self.radius * math.sin(-self.azimuth) * math.sin(self.inclination)])
        look_from = self.center
        look_to = self.center + offset
        self.c2w = torch.tensor(self.lookat(look_from, look_to), dtype=torch.float, device=self.device)

    def export_frame(self):
        pred_final = self.frame.detach().cpu().numpy().clip(0,1)*255
        cv2.imwrite("snapshot.png", pred_final[...,::-1])

    def render_one_frame(self, **kwargs):
        start = time.time()
        self.render(**kwargs)
        torch.cuda.synchronize()
        end = time.time()
        ms = end - start
        fps = 1. / ms
        ms *= 1000
        print(f"\r{ms:.2f} ms\t{fps:.2f} fps", end='')

    @torch.no_grad()
    def render(self, **kwargs):
        

        rays_d = torch.full((self.H, self.W, 3), -1, device=self.device, dtype=torch.float32)
        visitedTiles = torch.full((self.H, self.W, self.max_tracing_tile), -1, device=self.device, dtype=torch.int32)
        self.frame_diffuse = torch.full((self.H, self.W, 3), -1, device=self.device, dtype=torch.float32)
        inverse_near = torch.full((self.H, self.W), 1e-8, device=self.device, dtype=torch.float32)
        netIdxs = torch.full((self.H, self.W), -1, device=self.device, dtype=torch.int16)

        FASTRENDERING.compute_rays(self.intrinsic, self.c2w, self.num_thread, rays_d)

        FASTRENDERING.tracing_tiles(rays_d, self.tile_IndexMap, self.origin, self.tile_shape, self.scene_min_corner,
                                    self.scene_size, self.tile_size, self.num_thread, visitedTiles)

        FASTRENDERING.rendering_diffuse_octree_fp16(rays_d, visitedTiles, self.origin,
                                                self.tile2group, self.data_voxels, self.block_IndexMap,
                                                self.nodes_IndexMap, self.nodes_sampleFlag,
                                                self.voxels_start, self.centers, self.tile_size,
                                                self.voxel_size, self.sample_step, self.block_num_voxel,
                                                self.num_thread, self.trans_decay, self.early_stop_D, False,
                                                self.frame_diffuse,
                                                inverse_near, netIdxs)
        del visitedTiles


        self.large_inverse_near = inverse_near.clone()

        self.large_netIdx = netIdxs.clone()
        inverse_near = inverse_near[::self.reflection_scale,::self.reflection_scale]
        netIdxs = netIdxs[::self.reflection_scale,::self.reflection_scale].flatten()
        query_pixel_indices = torch.arange(0,self.downscale_numPixel,1, device=self.device, dtype=torch.int32)
        pixel_starts = torch.arange(0,self.downscale_numPixel,1, device=self.device, dtype=torch.int32)

        numNet = FASTRENDERING.sort_by_key(netIdxs, query_pixel_indices, pixel_starts)

        num_pixels_per_net =  torch.cat([pixel_starts[1:numNet], torch.ones(1, dtype=torch.int32, device=self.device)*self.downscale_numPixel], 0) - pixel_starts[:numNet]
        pixel_starts = torch.cat([pixel_starts[:numNet], torch.ones(1, dtype=torch.int32, device=self.device)*self.downscale_numPixel])
        block_starts = torch.cat([ torch.zeros(1, dtype=torch.int32, device=self.device), torch.cumsum( torch.ceil(num_pixels_per_net / self.max_pixels_per_block) , dim=0).int()], 0)
        netIdxs = netIdxs[:numNet]

        new_netIdxs = torch.empty(int(block_starts[-1]), device=self.device, dtype=torch.short)
        new_pixel_starts = torch.empty(int(block_starts[-1]), device=self.device, dtype=torch.int32)

        FASTRENDERING.assign_pixels(netIdxs, pixel_starts, block_starts, self.num_thread, new_netIdxs, new_pixel_starts)
        new_pixel_starts = torch.cat([new_pixel_starts, torch.ones(1, dtype=torch.int32, device=self.device)*self.downscale_numPixel], dim=0)

        self.boundary = torch.full((self.H,self.W,1), False, dtype=torch.bool, device=self.device)
        FASTRENDERING.compute_group_boundary(self.large_netIdx, self.boundary, self.ksize_boundary, self.num_thread*2)

        if self.render_mode == 2:
            self.frame = self.frame_diffuse 
            return 

        if self.render_mode == 4:
            self.frame = (1 / self.large_inverse_near[...,None] / self.max_depth).repeat(1,1,3)
            return 

        self.sample_far = 1. / inverse_near + self.far 

        inverse_bound = (1. / self.sample_far - inverse_near) / 63 # j/(N-1) * (1/f - 1/n)

        num_sample_per_step = 16 
        transparency = torch.ones((self.DH, self.DW, 1), device=self.device, dtype=torch.float32)
        self.frame_speculars = torch.zeros((self.DH, self.DW, 3), device=self.device, dtype=torch.float32)
        for step in range(0, 64, num_sample_per_step):
            sample_results = torch.zeros((self.DH, self.DW, num_sample_per_step, 4), device=self.device, dtype=torch.float32)
            samples = torch.full((self.DH, self.DW, num_sample_per_step, 3), -1, device=self.device, dtype=torch.float32)
            integrate_step = torch.zeros((self.DH, self.DW, num_sample_per_step), device=self.device, dtype=torch.float32)
            FASTRENDERING.inverse_z_sampling(self.origin, rays_d[::self.reflection_scale,::self.reflection_scale], inverse_near, inverse_bound, self.num_thread, 
                                            step, num_sample_per_step, samples, integrate_step)
            
            FASTRENDERING.inference_mlp(num_sample_per_step, int(block_starts[-1]), self.num_thread, 
                            self.inverse_far, self.early_stop_R, new_netIdxs, transparency,
                            query_pixel_indices, new_pixel_starts, 
                            self.group_centers, samples, rays_d[::self.reflection_scale,::self.reflection_scale], self.net_params, sample_results)

            FASTRENDERING.integrate_points(sample_results, integrate_step, self.num_thread*2, 
                                            self.early_stop_R, transparency, self.frame_speculars)

        self.frame_speculars = F.interpolate(self.frame_speculars[None,...].permute(0,3,1,2), 
                                             scale_factor=self.reflection_scale, mode='bilinear', 
                                             align_corners=True)[0].permute(1,2,0)

        if self.render_mode == 3:
            self.frame = self.frame_speculars 
            return 

        final = torch.clamp(self.frame_diffuse + self.frame_speculars, 0, 1)

        if self.enable_Unet:
            inputs = torch.cat([final[None,...].permute(0,3,1,2),self.large_inverse_near[None,...,None].permute(0,3,1,2), self.boundary[None,...].permute(0,3,1,2).float()],1)
            outputs = self.CNN(inputs)
            final = outputs[0].permute(1,2,0).clamp(0,1)

        self.frame = final 

        
    def memcpy(self):
        FASTRENDERING.frame_memcpy(self.frame_pointer, self.frame)

    def set_frame_pointer(self, frame_pointer):
        self.frame = None
        self.frame_pointer = frame_pointer
    
    
    

    

