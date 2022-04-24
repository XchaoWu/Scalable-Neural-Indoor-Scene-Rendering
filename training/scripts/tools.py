import numpy as np 
import cv2, os, yaml, re
from tqdm import tqdm  
import torch 
import imageio

def draw_AABB(centers, sizes, marks=[], color=(200,200,200)):
    """
    centers  N x 3 [cx, cy, cz]
    """
    init_coor = np.array(
        [[1,1,-1],[1,1,1],[-1,1,1],[-1,1,-1],
            [1,-1,-1],[1,-1,1],[-1,-1,1],[-1,-1,-1]], dtype=np.float32)

    init_face = np.array(
        [[0,1,4],[1,5,4],[0,4,7],[0,7,3],[2,3,7],[2,7,6],[1,6,5],[1,2,6],
        [0,2,1],[0,3,2],[4,5,6],[4,6,7]], dtype=np.int32) + 1

    vertex = []; face = []
    count = 0
    if isinstance(color, tuple): 
        for idx in tqdm(range(len(centers))): 
            center = centers[idx]
            size = sizes[idx]
            coords = init_coor.copy()
            coords = coords * (size / 2) + center
            
            colors = np.ones_like(coords) * 0.7
            if idx in marks:
                colors *= color
            coords = np.concatenate([coords, colors], -1)
            
            vertex += [coords]
            face += [init_face + count * 8]
            count += 1
        vertex = np.concatenate(vertex, 0)
        face = np.concatenate(face, 0)
        return vertex, face
    else:
        for idx in tqdm(range(len(centers))): 
            center = centers[idx]
            size = sizes[idx]
            coords = init_coor.copy()
            coords = coords * (size / 2) + center
            
            colors = np.ones_like(coords) * color[idx]
            coords = np.concatenate([coords, colors], -1)
            
            vertex += [coords]
            face += [init_face + count * 8]
            count += 1
        vertex = np.concatenate(vertex, 0)
        face = np.concatenate(face, 0)
        return vertex, face

def read_campara(path, return_shape=False):
    """
    read camera paras of Indoor Scene 
    format  

    index 
    fx fy cx cy
    width height near far 
    r11 r12 r13 t1
    r21 r22 r23 t2
    r31 r32 r33 t3 (camera2world)
    0   0   0   1
    """
    trans = lambda x:float(x)
    Ks = []
    C2Ws = []
    with open(path, 'r') as f:
        lines = f.readlines()
    
    for i in range(0,len(lines), 7):
        item = lines[i:i+7]
        name = item[0].strip()
        fx,fy,cx,cy = map(trans, re.split(r"\s+",item[1].strip()))
        width, height, near, far = map(trans, re.split(r"\s+",item[2].strip()))
        r11,r12,r13,t1 = map(trans, re.split(r"\s+",item[3].strip()))
        r21,r22,r23,t2 = map(trans, re.split(r"\s+",item[4].strip()))
        r31,r32,r33,t3 = map(trans, re.split(r"\s+",item[5].strip()))
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
        RT = np.array([[r11,r12,r13,t1],
                       [r21,r22,r23,t2],
                       [r31,r32,r33,t3]],dtype=np.float32)
        Ks += [K]
        C2Ws += [RT]
    Ks = np.stack(Ks, 0)
    C2Ws = np.stack(C2Ws, 0)
    print('\n=== Finish Loading camera ==')
    print(f'Ks shape: {Ks.shape}\tC2Ws shape: {C2Ws.shape}')
    if return_shape == False:
        return Ks, C2Ws
    else:
        return Ks, C2Ws, int(height), int(width)

# numpy  get rays
def get_rays_np(H, W, K, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0,2])/K[0,0], (j-K[1,2])/K[1,1], np.ones_like(i)], -1) 
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d   # H x W x 3

def get_rays_torch(H, W, K, c2w):
    device = K.device
    j,i = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    dirs = torch.stack([(i-K[0,2])/K[0,0], (j-K[1,2])/K[1,1], torch.ones_like(i, device=device)], -1) 
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].clone()[None,None,].repeat(H,W,1)
    return rays_o, rays_d

def line_scatter(A, B, step=0.01, colors=(255,255,255)):
    lam = np.arange(0,1+step,step)
    lam = np.stack([lam,lam,lam],axis=-1)
    C = (1 - lam) * A + lam * B 
    colors = np.ones_like(C) * colors
    return np.concatenate([C,colors], -1) # N x 3 

def camera_scatter(R, C, length=1, step=0.01):
    """
    R is world2camera rotation 
    """
    xs = line_scatter(C, C+R[0,:]*length, step, (255,0,0))
    ys = line_scatter(C, C+R[1,:]*length, step, (0,255,0))
    zs = line_scatter(C, C+R[2,:]*length, step, (0,0,255))

    return np.concatenate([xs,ys,zs],axis=0)

def cameras_scatter(Rs, Cs, length=2, step=0.01):
    scatters = []
    for R,C in zip(Rs, Cs):
        scatters += [camera_scatter(R, C, length, step)]
    return np.concatenate(scatters, 0)


def points2obj(out_path, points):
    """Converts point to obj format 
    """
    f = open(out_path, 'w')
    for item in points:
        f.write('v ' + ' '.join(list(map(lambda x:str(x), item))) + '\n')
    f.close()

def mesh2obj(out_path, vertex, face, color=None):
    f = open(out_path, 'w')
    for item in vertex:
        if color:
            f.write('v ' + ' '.join(list(map(lambda x:str(x), list(item)+list(color)))) + '\n')
        else:
            f.write('v ' + ' '.join(list(map(lambda x:str(x), list(item)))) + '\n')
    for item in face:
        f.write('f ' + ' '.join(list(map(lambda x:str(x), item))) + '\n')
    f.close()

def output_tile(out_path, centers, sizes, marks, color=(200,200,200), only_marks=False):
    if only_marks:
        centers = centers[marks]
        sizes = sizes[marks]
        vertex,face = draw_AABB(centers, sizes, marks=np.arange(len(centers)), color=color)
    else:
        vertex,face = draw_AABB(centers, sizes, marks=marks, color=color)
    mesh2obj(out_path, vertex, face)

def output_voxel(out_path, voxels, tile_centers, tile_sizes, voxel_size:float, th=1):

    has_volume = np.where(voxels[...,-1] >= th)
    tile_idxs, xs, ys, zs = has_volume
    colors = voxels[has_volume]
    colors = colors[:,:3]

    origin = tile_centers[tile_idxs] - tile_sizes[tile_idxs] / 2. + voxel_size / 2.

    centers = np.stack([xs,ys,zs], axis=-1)
    centers = origin + centers * voxel_size
    sizes = np.ones_like(centers) * voxel_size

    vertex, face = draw_AABB(centers, sizes, color=colors)
    mesh2obj(out_path, vertex, face)

def output_one_voxel(out_path, voxels, tile_center, tile_size, voxel_size:float, th=1):
    has_volume = np.where(voxels[...,-1] > th)
    xs, ys, zs = has_volume

    origin = tile_center - tile_size / 2.+ voxel_size / 2.

    centers = np.stack([xs,ys,zs], axis=-1)
    centers = origin + centers * voxel_size
    sizes = np.ones_like(centers) * voxel_size

    vertex, face = draw_AABB(centers, sizes)
    mesh2obj(out_path, vertex, face)

def output_T_voxel(out_path, T, tile_center, tile_size, voxel_size, th):
    
    T = T[1:-1,1:-1,1:-1]
    has_volume = np.where(T > th)
    # has_volume = np.where(node_flags != -1)
    xs, ys, zs = has_volume

    origin = tile_center - tile_size / 2.+ voxel_size / 2.
    centers = np.stack([xs,ys,zs], axis=-1)
    centers = origin + centers * voxel_size
    sizes = np.ones_like(centers) * voxel_size

    vertex, face = draw_AABB(centers, sizes)
    mesh2obj(out_path, vertex, face)


def output_node_voxel_colored(out_path, voxels, tile_center, tile_size, voxel_size, node_flags):

    # node_flags = node_flags[1:-1,1:-1,1:-1]
    # voxels = voxels[1:-1,1:-1,1:-1]
    has_volume = np.where(node_flags == 1)
    # has_volume = np.where(node_flags != -1)
    xs, ys, zs = has_volume
    sigma = voxels[xs, ys, zs, -1]
    alpha = 1 - np.exp(-sigma)
    print(alpha.min(), alpha.max())

    origin = tile_center - tile_size / 2.+ voxel_size / 2.
    centers = np.stack([xs,ys,zs], axis=-1)
    centers = origin + centers * voxel_size
    sizes = np.ones_like(centers) * voxel_size

    colors = np.ones_like(centers) * alpha[...,None]

    vertex, face = draw_AABB(centers, sizes, colors)
    mesh2obj(out_path, vertex, face)

def mesh2nodes(obj_path, num_voxel, voxel_size, tile_centers):
    with open(obj_path, 'r') as f:
        lines = f.readlines()
    vertices = []
    for line in lines:
        line = line.strip().split(' ')
        if line[0] == 'v':
            vertices.append([float(line[1]), float(line[2]), float(line[3])])
        elif line[0] == 'f':
            break 
    vertices = np.array(vertices).reshape(-1,8,3)
    voxel_center = np.mean(vertices, axis=1)

    tile_size = num_voxel * voxel_size 

    nodes_list = []
    for tile_center in tqdm(tile_centers):
        tile_corner = tile_center - tile_size / 2. 
        loc = (voxel_center - tile_corner) // voxel_size # N x 3 
        loc = loc.astype(np.int32)
        valid = np.all(loc >= 0, axis=-1)
        valid = valid & np.all(loc < num_voxel, axis=-1)
        loc = loc[valid]

        temp = np.ones((num_voxel+2, num_voxel+2, num_voxel+2), dtype=np.int16) * -1 
        nodes = np.ones((num_voxel, num_voxel, num_voxel), dtype=np.int16) * -1 
        if loc.shape[0] > 0:
            nodes[(loc[:,0], loc[:,1], loc[:,2])] = 1
        temp[1:-1,1:-1,1:-1] = nodes
        nodes_list.append(temp)
    return np.array(nodes_list)


def points2nodes(obj_path, num_voxel, voxel_size, tile_centers):
    with open(obj_path, 'r') as f:
        lines = f.readlines()
    vertices = []
    for line in lines:
        line = line.strip().split(' ')
        if line[0] == 'v':
            vertices.append([float(line[1]), float(line[2]), float(line[3])])
        elif line[0] == 'f':
            break 
    voxel_center = np.array(vertices)

    tile_size = num_voxel * voxel_size 

    nodes_list = []
    for tile_center in tqdm(tile_centers):
        tile_corner = tile_center - tile_size / 2. 
        loc = (voxel_center - tile_corner) // voxel_size # N x 3 
        loc = loc.astype(np.int32)
        valid = np.all(loc >= 0, axis=-1)
        valid = valid & np.all(loc < num_voxel, axis=-1)
        loc = loc[valid]

        temp = np.ones((num_voxel+2, num_voxel+2, num_voxel+2), dtype=np.int16) * -1 
        nodes = np.ones((num_voxel, num_voxel, num_voxel), dtype=np.int16) * -1 
        if loc.shape[0] > 0:
            nodes[(loc[:,0], loc[:,1], loc[:,2])] = 1
        temp[1:-1,1:-1,1:-1] = nodes
        nodes_list.append(temp)
    return np.array(nodes_list)

def output_node_batch_point(out_path, tile_center, tile_size, voxel_size, node_flags):
    
    all_centers = []
    for c, node in zip(tile_center, node_flags):
        node = node[1:-1,1:-1,1:-1]
        has_volume = np.where(node == 1)
        xs, ys, zs = has_volume
        origin = c - tile_size / 2.+ voxel_size / 2.
        center = np.stack([xs,ys,zs], axis=-1)
        center = origin + center * voxel_size
        all_centers.append(center)
    all_centers = np.concatenate(all_centers, 0)
    # sizes = np.ones_like(all_centers) * voxel_size
    colors = np.ones_like(all_centers) * (255,0,0)
    pts = np.concatenate([all_centers, colors], -1)
    points2obj(out_path, pts)

    # vertex, face = draw_AABB(all_centers, sizes)
    # mesh2obj(out_path, vertex, face)

def output_node_batch(out_path, tile_center, tile_size, voxel_size, node_flags):
    
    all_centers = []
    for c, node in zip(tile_center, node_flags):
        node = node[1:-1,1:-1,1:-1]
        has_volume = np.where(node == 1)
        xs, ys, zs = has_volume
        origin = c - tile_size / 2.+ voxel_size / 2.
        center = np.stack([xs,ys,zs], axis=-1)
        center = origin + center * voxel_size
        all_centers.append(center)
    all_centers = np.concatenate(all_centers, 0)
    sizes = np.ones_like(all_centers) * voxel_size

    vertex, face = draw_AABB(all_centers, sizes)
    mesh2obj(out_path, vertex, face)

def output_node_voxel(out_path, tile_center, tile_size, voxel_size, node_flags):

    node_flags = node_flags[1:-1,1:-1,1:-1]
    has_volume = np.where(node_flags == 1)
    # has_volume = np.where(node_flags != -1)
    xs, ys, zs = has_volume

    origin = tile_center - tile_size / 2.+ voxel_size / 2.
    centers = np.stack([xs,ys,zs], axis=-1)
    centers = origin + centers * voxel_size
    sizes = np.ones_like(centers) * voxel_size

    vertex, face = draw_AABB(centers, sizes)
    mesh2obj(out_path, vertex, face)

def output_node_voxel_v2(out_path, tile_center, tile_size, voxel_size, node_flags):
    # has_volume = np.where(node_flags == 1)
    node_flags = node_flags[1:-1,1:-1,1:-1]
    has_volume = np.where(node_flags != -1)
    xs, ys, zs = has_volume

    origin = tile_center - tile_size / 2.+ voxel_size / 2.
    centers = np.stack([xs,ys,zs], axis=-1)
    centers = origin + centers * voxel_size
    sizes = np.ones_like(centers) * voxel_size

    vertex, face = draw_AABB(centers, sizes)
    mesh2obj(out_path, vertex, face)

def load_preprocess(path):
    get_array = lambda x: np.load(os.path.join(path,x))
    # voxels = get_array("overlapVoxels.npy")
    # voxels = get_array("voxels.npy")
    voxels = None
    centers = get_array("centers.npy")
    IndexMap = get_array("IndexMap.npy")
    scene_min_corner = get_array("scene_min_corner.npy")
    tile_shape = get_array("tile_shape.npy")
    BConFaceIdx = get_array("BConFaceIdx.npy")
    BConFaceNum = get_array("BConFaceNum.npy")
    VisImg = get_array("VisImg.npy")
    SparseToDense = get_array("SparseToDense.npy")
    return voxels, centers, IndexMap, SparseToDense, BConFaceIdx, \
           BConFaceNum, VisImg, scene_min_corner, tile_shape


def generate_video(save_path, img_list, fps=20):
    # img_list = [x.astype(np.uint8) for x in img_list]
    # imageio.mimwrite(save_path, img_list, fps=fps)
    out = cv2.VideoWriter(save_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (img_list[0].shape[1],img_list[0].shape[0]))
    for img in img_list:
        out.write(img[...,::-1].astype(np.uint8))
    out.release()

def generate_gif(save_path, img_list, fps):
    img_list = [x.astype(np.uint8) for x in img_list]
    imageio.mimsave(save_path, img_list, fps=fps)

def save_img_list(save_path, img_list):
    img_list = [x.astype(np.uint8) for x in img_list]
    for idx, x in enumerate(img_list):
        cv2.imwrite(os.path.join(save_path, f"{idx}.png"), x[...,::-1])
