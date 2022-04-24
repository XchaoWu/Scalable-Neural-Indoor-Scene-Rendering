import numpy as np 
import torch 
import torch.nn as nn 

class PinholeCamera:
    def __init__(self, height, width, C2W, device, **kwargs):
        self.height = height 
        self.width = width 

        if 'near' in kwargs.keys():
            self.near = kwargs['near'] 
            self.K = self.compute_intrinsic()
        else:
            self.K = kwargs['K']
            self.fx = self.K[0,0]
            self.fy = self.K[1,1]
            self.cx = self.K[0,2]
            self.cy = self.K[1,2]

        self.device = device 

        self.C2W = torch.from_numpy(C2W).to(device).float()
    
    def move_x(self, step):
        self.C2W[:,3] += self.C2W[:3,0] * step   
    
    def move_y(self, step):
        self.C2W[:,3] += self.C2W[:3,1] * step  

    def move_z(self, step):
        self.C2W[:,3] += self.C2W[:3,2] * step 

    def set_rotateY(self, theta):
        self.ry = torch.from_numpy(self.Ry(theta)).to(self.device).float()
        self._ry = torch.from_numpy(self.Ry(-theta)).to(self.device).float()
    
    def set_rotateX(self, theta):
        self.rx = torch.from_numpy(self.Rx(theta)).to(self.device).float()
        self._rx = torch.from_numpy(self.Rx(-theta)).to(self.device).float()
    
    def rotate_y(self, sign):
        if sign > 0:
            self.C2W[:3,:3] = self.ry @ self.C2W[:3,:3]
        else:
            self.C2W[:3,:3] = self._ry @ self.C2W[:3,:3]

    def rotate_x(self, sign):
        if sign > 0:
            self.C2W[:3,:3] = self.rx @ self.C2W[:3,:3]
        else:
            self.C2W[:3,:3] = self._rx @ self.C2W[:3,:3]
    
    def compute_intrinsic(self):
        self.fx = self.near * (self.width-1)/2. 
        self.fy = self.near * (self.height-1)/2. 
        self.fy = self.fx 
        self.cx = (self.width-1)/2. 
        self.cy = (self.height-1)/2. 
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0,0,1]])
    
    def get_rays(self):
        j,i = torch.meshgrid(torch.arange(self.height, device=self.device), 
                             torch.arange(self.width, device=self.device))
        dirs = torch.stack([(i-self.cx)/self.fx, (j-self.cy)/self.fy, 
                             torch.ones_like(i, device=self.device)], -1) 
        rays_d = torch.sum(dirs[..., None, :] * self.C2W[:3, :3], -1).reshape(-1,3)
        rays_o = self.C2W[:3, -1].clone().reshape(-1,3).repeat(rays_d.shape[0],1)
        return rays_o.float(), rays_d.float() 
    

    @staticmethod
    def compute_C2W(z_axis, up, focus, distance):
        normalize = lambda x: x / np.linalg.norm(x)
        x_axis = np.cross(up, z_axis)
        y_axis = np.cross(z_axis, x_axis)
        x_axis = normalize(x_axis)
        y_axis = normalize(y_axis)
        z_axis = normalize(z_axis)
        C = focus - z_axis * distance 
        # 3 x 4 
        return np.stack([x_axis,y_axis,z_axis,C],axis=-1)
    
    @staticmethod
    def Rx(theta):
        R = np.array([[1, 0, 0], 
                      [0, np.cos(theta), -np.sin(theta)], 
                      [0, np.sin(theta), np.cos(theta)]], 
                      dtype=np.float32)
        return R 

    @staticmethod
    def Ry(theta):
        R = np.array([[np.cos(theta), 0, np.sin(theta)], 
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]], 
                        dtype=np.float32)
        return R  

    @staticmethod
    def Rz(theta):
        R = np.array([[np.cos(theta), -np.sin(theta), 0], 
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]], 
                        dtype=np.float32)
        return R 



def normalize(x):
    return x / np.linalg.norm(x)

def view_matrix(z_axis, up, C):
    z_axis = normalize(z_axis)
    x_axis = normalize(np.cross(up,z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.stack([x_axis,y_axis,z_axis,C],axis=-1)

def compute_captured_camera_path(height, width, C2Ws, Ks, device, **kwargs):
    cameras = []
    Ks = np.array(Ks) * kwargs['scale']
    Ks[:,2,2] = 1
    for K, C2W in zip(Ks, C2Ws):
        cam = PinholeCamera(height, width, C2W, device, K = K)
        cameras.append(cam)
    return cameras

def compute_keypoint_camera_path(height, width, num, device, scale, **kwargs):
    cameras = []
    key_C2Ws = kwargs['key_C2Ws'] # N x 3 x 4 
    key_K = kwargs['key_K'] * scale
    height = int(height * scale)
    width = int(width * scale)
    key_K[2,2] = 1

    C2W1 = key_C2Ws[0]
    
    for idx, C2W2 in enumerate(key_C2Ws[1:]):
        up = 0.5 * C2W1[:,1] + 0.5 * C2W2[:,1]
        for sidx, step in enumerate(np.linspace(0,1,num)):
            center = C2W1[:,3] * (1-step) + C2W2[:,3] * step
            z = C2W1[:,2] * (1-step) + C2W2[:,2] * step
            up = C2W1[:,1] * (1-step) + C2W2[:,1] * step
            x = np.cross(up, z)
            y = np.cross(z,x)
            x = x / np.linalg.norm(x)
            y = y / np.linalg.norm(y)
            z = z / np.linalg.norm(z)
            C2W = np.stack([x,y,z,center],axis=-1)
            cam = PinholeCamera(height, width, C2W, device, K = key_K)
            cameras.append(cam)
        C2W1 = C2W2
    return cameras


def compute_inside_camera_path(height, width, num, device, scale, **kwargs):
    cameras = []
    
    C2W1 = kwargs['C2W1']
    C2W2 = kwargs['C2W2']
    K = kwargs['K'] * scale
    K[2,2] = 1

    height = int(height * scale)
    width = int(width * scale)

    up = 0.5 * C2W1[:,1] + 0.5 * C2W2[:,1]
    for step in np.linspace(0,1,num):
        center = C2W1[:,3] * (1-step) + C2W2[:,3] * step
        z = C2W1[:,2] * (1-step) + C2W2[:,2] * step
        x = np.cross(up, z)
        y = np.cross(z,x)
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = z / np.linalg.norm(z)
        C2W = np.stack([x,y,z,center],axis=-1)
        cam = PinholeCamera(height, width, C2W, device, K = K)
        cameras.append(cam)
    return cameras


def compute_spiral_camera_path(height, width, near, up, focus, distance, num, device, **kwargs):
    cameras = []
    z_axis = kwargs['face']
    base_c2w = PinholeCamera.compute_C2W(z_axis, up, focus, distance)
    step = 2 / num
    for theta in np.arange(0, 2, step):
        theta = theta * np.pi 
        center = np.dot(base_c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*kwargs['zrate']), 1.]) * kwargs['rads'])
        z = focus - center
        C2W = PinholeCamera.compute_C2W(z, up, focus, distance)
        cam = PinholeCamera(height, width, C2W, device, near=near)
        cameras.append(cam)
    return cameras 

def compute_spiral_camera_path_LLFF(height, width, C2Ws, num, device, scale, **kwargs):
    cameras = []

    base_center = np.mean(C2Ws[:,:,3],axis=0) # 3 
    base_z_axis = np.mean(C2Ws[:,:,2],axis=0) # 3 
    up = np.mean(C2Ws[:,:,1],axis=0) # 3 

    base_c2w = view_matrix(base_z_axis, up, base_center)
    tt = C2Ws[:,:3,3]
    rads = np.percentile(np.abs(tt-base_center), 90, 0)
    rads = np.array(list(rads) + [1.])
    rads[2] = 0

    focal = kwargs['focal']
    cx = height / 2 
    cy = width / 2

    K = np.array([focal*scale, 0, cx*scale, 0, focal*scale, cy*scale, 0, 0, 1],dtype=np.float32).reshape(3,3)

    height = int(height * scale)
    width = int(width * scale)
    step = 2 / num
    for theta in np.arange(0, 2, step):
        theta = theta * np.pi
        center = np.dot(base_c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*0.5), 1.]) * rads)
        z_axis = normalize(center - np.dot(base_c2w[:3,:4], np.array([0,0,-focal,1.])))
        C2W = view_matrix(z_axis, up, center)
        cam = PinholeCamera(height, width, C2W, device, K=K)
        cameras.append(cam)
    return cameras 
        


def compute_360_camera_path(height, width, near, up, focus, distance, num, device, **kwargs):
    cameras = []
    start_z = kwargs['face']
    step = 2 / num
    for theta in np.arange(0, 2, step):
        Ry = PinholeCamera.Ry(theta*np.pi)
        z_axis = Ry @ start_z 
        C2W = PinholeCamera.compute_C2W(z_axis, up, focus, distance)
        cam = PinholeCamera(height, width, C2W, device, near=near)
        cameras.append(cam)
    return cameras 



    
    

    