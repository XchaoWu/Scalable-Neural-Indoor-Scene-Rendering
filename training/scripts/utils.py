from genericpath import isdir
import numpy as np 
from tqdm import tqdm 
import yaml 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import cv2,os,sys 
import imageio
from glob import glob 
sys.path.append('../')
from src import (
    Sample_uniform,
    Sample_bg,
    Sample_sparse,
    Sample_reflection, 
    dilate_boundary,
    compute_grid)
import random 



"""configuration parse 
"""
class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict2obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dict2obj(v)
    return d

def parse_yaml(path):
    with open(path, 'r') as f:
        cfg = dict2obj(yaml.full_load(f.read()))
    return cfg


def depth_smoothess_loss(depth, color, gamma):
    """
    depth H x W x 1
    color H x W x 3 
    """
    # H-1 x W x 1 
    diff_d_h = torch.abs(depth[1:,...] - depth[:-1,...]) 
    # H-1 x W x 1
    diff_c_h = torch.mean(torch.abs(color[1:,...] - color[:-1, ...]), dim=-1, keepdim=True)

    # H-1 x W x 1
    diff_h = torch.mean(diff_d_h * torch.exp(-gamma * diff_c_h))

    diff_d_w = torch.abs(depth[:,1:,...] - depth[:,:-1,...]) 
    diff_c_w = torch.mean(torch.abs(color[:,1:,...] - color[:,:-1, ...]), dim=-1, keepdim=True)
    diff_w = torch.mean(diff_d_w * torch.exp(-gamma * diff_c_w))
    return diff_h + diff_w 



def get_groups(group_txt_dir):
    files = glob(os.path.join(group_txt_dir, "*.txt"))
    groups = []
    groups_name = []

    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            lines = [item.strip() for item in lines]
            lines = [item for item in lines if item != '']
            if len(lines) == 0:
                continue 
            tileIdxs = [int(item.strip()) for item in lines]
            if len(tileIdxs) == 0:
                continue 
            groups.append(tileIdxs)
        name = os.path.splitext(os.path.basename(file))[0]
        groups_name.append(name)

    temp = []
    for item in groups:
        temp += item 
    max_tileIdx = np.max(temp)
    tile2group = np.ones((max_tileIdx+1), dtype=np.int32) * -1 
    for idx,item in enumerate(groups):
        for t in item:
            tile2group[t] = idx

    return groups, groups_name, tile2group
    

def load_ignore_v2(ignore_path):
    ignore = []
    try:
        f = open(ignore_path, 'r')
    except:
        pass
    else:
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.strip().split(' ')
            if len(line) == 1:
                ignore += [int(line[0])]
            elif len(line) == 2:
                ignore += list(np.arange(int(line[0]), int(line[1])))
    return ignore

def search_tiles_by_groups(pretrained_dir, groups_names):
    tileIdx_list = []
    for group_name in groups_names:
        print(group_name)
        tile_files = glob(os.path.join(pretrained_dir, group_name, "tile-*"))
        tile_files = [item for item in tile_files if os.path.isdir(item)]
        tileIdx_list += [int(os.path.split(item)[-1].split('-')[1]) for item in tile_files]
    return tileIdx_list

def search_pretrained_tiles(pretrained_dir, tiles_idx):
    sorted_func = lambda x:int(os.path.splitext(os.path.basename(x))[0].split('-')[2][1:])


    tileIdx_list = []
    file_list = glob(os.path.join(pretrained_dir, '*'))
    file_list = [item for item in file_list if os.path.isdir(item)]


    voxels = []
    nodes = []

    print(tiles_idx)

    for tileIdx in tiles_idx:
        find_flag = False
        for file in file_list:
            tile_file = os.path.join(file, f"tile-{tileIdx}")
            if os.path.isdir(tile_file):
                print(f"tile {tileIdx} in file {tile_file}")
                error_tiles = []
                try:
                    with open(os.path.join(file, "errorTiles.txt"), 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = int(line.strip().split('\t')[1].split(' ')[1])
                            error_tiles.append(line)
                except:
                    pass 
                if tileIdx not in error_tiles:
                    find_flag = True 
                else:
                    print(f"tile {tileIdx} is invalid tile")
                break 
            else:
                continue 
        if find_flag:
            voxel_path = glob(os.path.join(tile_file, f'Voxel*.npy'))
            voxel_path.sort(key=sorted_func)
            voxel_path = voxel_path[-1]
            nodes_path = glob(os.path.join(tile_file, f'Node*.npy'))
            nodes_path.sort(key=sorted_func)
            nodes_path = nodes_path[-1]
            voxels.append(np.load(voxel_path))
            nodes.append(np.load(nodes_path))
            tileIdx_list.append(tileIdx)
        else:
            print(f"Not found! tile {tileIdx} pretrained")

    return np.stack(voxels, 0), np.stack(nodes, 0), np.array(tileIdx_list)

# if __name__ == '__main__':
#     voxels, nodes = search_pretrained_tiles("/home/yons/4TB/data/wangchi_scene/logs/scene3", [2306, 460])
#     print(voxels.shape, nodes.shape)

def get_neighbor_tiles(tileIdxs_list, SparseToDense, IndexMap, tile_shape):

    xs,ys,zs = np.meshgrid(np.arange(-1,2), np.arange(-1,2), np.arange(-1,2))
    offset = np.stack([xs,ys,zs], axis=-1).reshape(-1,3)
    offset = np.concatenate([offset[:13],offset[14:]], axis=0)

    nei_tiles = []
    for tileIdx in tileIdxs_list:
        denseIdx = SparseToDense[tileIdx]
        x = denseIdx // (tile_shape[1] * tile_shape[2])
        y = (denseIdx % (tile_shape[1] * tile_shape[2])) // tile_shape[2]
        z = (denseIdx % (tile_shape[1] * tile_shape[2])) % tile_shape[2]
        ref = np.array([x,y,z])
        neis = ref + offset 
        valid = (neis[...,0] >= 0) & (neis[...,0] < tile_shape[0])
        valid = valid & (neis[...,1] >= 0) & (neis[...,1] < tile_shape[1])
        valid = valid & (neis[...,2] >= 0) & (neis[...,2] < tile_shape[2])
        neis = neis[valid].reshape(-1,3)
        nei_denseIdx = neis[:,0] * tile_shape[1] * tile_shape[2] + neis[:,1] * tile_shape[2] + neis[:,2]
        nei_tileIdxs = IndexMap[nei_denseIdx]
        nei_tileIdxs = [item for item in nei_tileIdxs if item != -1]
        nei_tiles += [item for item in nei_tileIdxs if item not in tileIdxs_list]

    return np.array(list(tileIdxs_list) + list(set(nei_tiles)))

def get_neighborNtiles(tileIdxs_list, SparseToDense, IndexMap, tile_shape, dilate_size = 1):
    for i in range(dilate_size):
        tileIdxs_list = get_neighbor_tiles(tileIdxs_list, SparseToDense, IndexMap, tile_shape)
    return tileIdxs_list


def split_filename(path):
    for item in path.split('/')[::-1]:
        if item != '':
            return item 
    return None 

def build_training_info(path, numGPU, sorted=True, inverse=True):
    file_list = glob(os.path.join(path, '*.txt'))
    if sorted:
        file_list.sort(key=lambda x:int(os.path.splitext(os.path.basename(x))[0][5:]))
        if inverse:
            file_list = file_list[::-1]
    temp = []
    for i in range(numGPU):
        temp += file_list[i::numGPU] 
    file_list = temp 
    info = []
    for file in file_list:
        with open(file, 'r') as f:
            lines = f.readlines()
        lines = [int(item.strip()) for item in lines] 
        info += [(file, lines)]

    return info 


class Img2Gradient(nn.Module):
    """ compute the gradient of img by pytorch
    """
    def __init__(self, in_channel):
        super(Img2Gradient, self).__init__()
        kh = torch.tensor([[1.,0.,-1.],
                            [2.,0.,-2.],
                            [1.,0.,-1.]], dtype=torch.float32, requires_grad=False)
        kv = torch.tensor([[1.,2., 1.],
                           [0.,0., 0.],
                           [-1.,-2.,-1.]], dtype=torch.float32, requires_grad=False)
        kh = kh.view(1,1,3,3).repeat(1,in_channel,1,1) / 3.
        kv = kv.view(1,1,3,3).repeat(1,in_channel,1,1) / 3.
        self.register_buffer("KH", kh)
        self.register_buffer("KV", kv)
    def forward(self,x):
        xh = F.conv2d(x, self.KH)
        xv = F.conv2d(x, self.KV)
        return (torch.abs(xv) + torch.abs(xh)) / 2.0


def GradLoss(out, gt):
    """
    HWC
    """
    oy = out[1::3, :, :] - out[0::3, :, :]
    ox = out[2::3, :, :] - out[0::3, :, :]
    gy = gt[1::3, :, :] - gt[0::3, :, :]
    gx = gt[2::3, :, :] - gt[0::3, :, :]
    return torch.mean(torch.abs(ox-gx)) + torch.mean(torch.abs(oy-gy))

def Mask_L1Loss(x1, x2, mask):
    item = torch.sum(mask)
    if item != 0:
        return torch.sum(torch.abs(x1-x2) * mask) / item
    else:
        return 0 

def Mask_MSELoss(x1, x2, mask):
    item = torch.sum(mask)
    if item != 0:
        return torch.sum(torch.abs(x1-x2)**2 * mask) / item
    else:
        return 0

def compute_hard_mining(x: torch.Tensor, top_rate=0.2):
    N,H,W = x.shape
    flat_x = x.reshape(N, H*W)
    value, idx = flat_x.sort(-1, descending=True) #  N x (HW)
    top_num = int(H*W * top_rate)
    
    idx = idx[:,:top_num]
    t = torch.arange(N).reshape(-1,1).repeat(1,idx.shape[1]).cuda()
    k = torch.cat([t[...,None],idx[...,None]],-1).reshape(-1,2).transpose(1,0)

    masks = torch.zeros(N,H*W)
    masks[k[0],k[1]] = 1.
    masks = masks.reshape(N,H,W)
    masks = masks.cuda()
    return masks, top_num


def binary_loss(alpha):
    item = torch.exp(-(alpha-0.5) * 10)
    return torch.mean(item / ((1. + item)**2))

def hard_mining_loss(pred, target, ratio=0.2):
    """
    pred NCHW
    target NCHW
    """
    mask,top_num = compute_hard_mining(torch.mean(torch.abs(pred - target),1),ratio)
    return torch.sum(torch.mean(torch.abs(pred - target),1) * mask) / top_num
    

def sparsity_loss(sigma, lam):
    return torch.mean(1. - torch.exp(-lam*sigma))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_pretrained_tiles(root_dir, epoch=None):

    out = {'group_name': split_filename(root_dir)}

    error_tiles = []
    try:
        with open(os.path.join(root_dir, "errorTiles.txt"), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = int(line.strip().split('\t')[1].split(' ')[1])
                error_tiles.append(line)
    except:
        pass 
    print(root_dir)

    try:
        group_center = np.load(os.path.join(root_dir, "group_center.npy"))
    except:
        pass 
    else:
        out['group_center'] = group_center

    sorted_func = lambda x:int(os.path.splitext(os.path.basename(x))[0].split('-')[2][1:])
    if epoch:
        coeffi_path = os.path.join(root_dir, f'Coeffi-TShared-E{epoch}.pt')
        model_path = os.path.join(root_dir, f'Model-TShared-E{epoch}.pt')
    else:
        file_list = glob(os.path.join(root_dir, f'Coeffi*.pt'))
        file_list.sort(key=sorted_func)
        coeffi_path = file_list[-1] 
        file_list = glob(os.path.join(root_dir, f'Model*.pt'))
        file_list.sort(key=sorted_func)
        model_path = file_list[-1]
    

    voxels_path_list = []
    nodes_path_list = []
    tileIdx_list = []

    file_path = glob(os.path.join(root_dir, "tile-*"))
    file_path.sort(key=lambda x: int(x.split('/')[-1].split('-')[1]))

    for path in file_path:

        tileIdx = int(path.split('/')[-1].split('-')[-1])

        if tileIdx in error_tiles:
            continue 

        if epoch:
            nodes_path = glob(os.path.join(path, f'Node*E{epoch}.npy'))
            voxel_path = glob(os.path.join(path, f'Voxel*E{epoch}.npy'))
        else:
            nodes_path = glob(os.path.join(path, f'Node*.npy'))
            nodes_path.sort(key=sorted_func)
            voxel_path = glob(os.path.join(path, f'Voxel*.npy'))
            voxel_path.sort(key=sorted_func)
            nodes_path = nodes_path[-1]
            voxel_path = voxel_path[-1]
        
        tileIdx_list.append(tileIdx)
        voxels_path_list.append(voxel_path)
        nodes_path_list.append(nodes_path)
    
    assert len(tileIdx_list) == len(voxels_path_list)
    assert len(tileIdx_list) == len(nodes_path_list)


    if len(tileIdx_list) == 0:
        print(f"[WARN] number of valid tiles is 0:\n {root_dir}")
        return None
    else:
        out['tileIdxs'] = tileIdx_list
        out['coeffi'] = coeffi_path
        out['model'] = model_path
        out['voxels'] = voxels_path_list
        out['nodes'] = nodes_path_list
        return out 

def get_group_tiles(root_dir, epoch=None):

    out = {'group_name': split_filename(root_dir)}

    error_tiles = []
    try:
        with open(os.path.join(root_dir, "errorTiles.txt"), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = int(line.strip().split('\t')[1].split(' ')[1])
                error_tiles.append(line)
    except:
        pass 

    try:
        group_center = np.load(os.path.join(root_dir, "group_center.npy"))
    except:
        pass 
    else:
        out['group_center'] = group_center

    sorted_func = lambda x:int(os.path.splitext(os.path.basename(x))[0].split('-')[2][1:])
    if epoch:
        coeffi_path = os.path.join(root_dir, f'Coeffi-TShared-E{epoch}.pt')
    else:
        file_list = glob(os.path.join(root_dir, f'Coeffi*.pt'))
        if len(file_list) == 0:
            return None 
        file_list.sort(key=sorted_func)
        coeffi_path = file_list[-1] 
        # if sorted_func(coeffi_path) != 50:
        #     return None

    voxels_path_list = []
    nodes_path_list = []
    tileIdx_list = []

    file_path = glob(os.path.join(root_dir, "tile-*"))
    for path in file_path:
        if os.path.isdir(path) is False:
            continue
        tileIdx = int(path.split('/')[-1].split('-')[-1])
        if tileIdx in error_tiles:
            continue 
        if epoch:
            nodes_path = glob(os.path.join(path, f'Node*E{epoch}.npy'))[0]
            voxel_path = glob(os.path.join(path, f'Voxel*E{epoch}.npy'))[0]
        else:
            nodes_path = glob(os.path.join(path, f'Node*.npy'))
            nodes_path.sort(key=sorted_func)
            voxel_path = glob(os.path.join(path, f'Voxel*.npy'))
            voxel_path.sort(key=sorted_func)
            nodes_path = nodes_path[-1]
            voxel_path = voxel_path[-1]
            # if sorted_func(nodes_path) != 50:
            #     return None
            # if sorted_func(voxel_path) != 50:
            #     return None
        tileIdx_list.append(tileIdx)
        voxels_path_list.append(voxel_path)
        nodes_path_list.append(nodes_path)
    
    assert len(tileIdx_list) == len(voxels_path_list)
    assert len(tileIdx_list) == len(nodes_path_list)

    if len(tileIdx_list) == 0:
        print(f"[WARN] number of valid tiles is 0:\n {root_dir}")
        return None
    else:
        print(f"successfully loading {root_dir}")
        out['tileIdxs'] = tileIdx_list
        out['coeffi'] = coeffi_path
        out['voxels'] = voxels_path_list
        out['nodes'] = nodes_path_list
        return out 
        

def get_trained_tiles(root_dir, epoch=None):
    file_list = glob(os.path.join(root_dir, '*'))
    file_list = [item for item in file_list if os.path.isdir(item)]


    group_file_list = []
    others = []
    for item in file_list:
        file_name = os.path.split(item)[1]
        if 'group' in file_name:
            group_file_list.append(item)
        else:
            others.append(item)

    group_file_list.sort(key=lambda x: int(os.path.split(x)[1][5:]) )
    
    file_list = group_file_list + others

    info = []
    for file in file_list:
        res = get_group_tiles(file, epoch)
        if res:
            res['dir'] = file
            info.append(res)
    return info 


def extract_path(root_dir, epoch=None):
    sorted_func = lambda x:int(os.path.splitext(os.path.basename(x))[0].split('-')[2][1:])
    voxels_path_list = []
    nodes_path_list = []
    models_path_list = []

    exists = []

    try:
        if epoch:
            shared_path = os.path.join(root_dir, f'Coeffi-TShared-E{epoch}.pt')
        else:
            file_list = glob(os.path.join(root_dir, f'Coeffi*.pt'))
            file_list.sort(key=sorted_func)
            shared_path = file_list[-1]
    except:
        shared_path = None 
        print("no shared_path")
    else:
        print(f"shared_path {shared_path}")
    
    file_path = glob(os.path.join(root_dir, "tile-*"))
    
    # print(file_path)

    for path in file_path:

        if epoch:
            nodes_path = glob(os.path.join(path, f'Node*E{epoch}.npy'))
            # nodes_path_list.append(os.path.join(path, f'Node*E{{epoch}}.npy'))
        else:
            nodes_path = glob(os.path.join(path, f'Node*.npy'))

        if len(nodes_path) == 0:
            continue 

        nodes_path.sort(key=sorted_func)

        if epoch:
            voxel_path = glob(os.path.join(path, f'Voxel*E{epoch}.npy'))
        else:
            voxel_path = glob(os.path.join(path, f'Voxel*.npy'))

        voxel_path.sort(key=sorted_func)

        # model_path = glob(os.path.join(path, f'Coeffi*.pt'))
        # try:
        #     model_path.sort(key=sorted_func)
        #     models_path_list.append(model_path[-1])
        # except:
        #     pass

        nodes_path_list.append(nodes_path[-1])
        voxels_path_list.append(voxel_path[-1])
        tileIdx = int(path.split('/')[-1].split('-')[-1])
        exists.append(tileIdx)
    return exists, voxels_path_list, nodes_path_list, models_path_list, shared_path

def extrac_MLP_matrix(model):

    weights = []
    bias = []

    for item in model.mlp:
        if isinstance(item, nn.Linear):
            weights.append(item.weight)
            bias.append(item.bias)
    
    return weights, bias 


def extrac_MLP_para(path):
    model_dict = torch.load(path)
    weights = []
    bias = []
    for key in model_dict.keys():
        if 'weight' in key:
            weights.append(model_dict[key])
        elif 'bias' in key:
            bias.append(model_dict[key])
    return weights, bias 


@torch.no_grad()
def trilinear_weight(x):
    device = x.device
    # 8 x 3
    base = torch.tensor([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                         [1,1,0],[1,0,1],[0,1,1],[1,1,1]],
                         dtype = torch.int32,
                         device = device)
    x0 = x.int() # B x N x 3 
    # B x N x 8 x 3 
    x_nei = x0[...,None,:] + base[None,None,...]
    x_weight = torch.prod(torch.abs(x[...,None,:] - x_nei.float()), dim=-1)
    return x_nei, x_weight[..., None]

@torch.no_grad()
def getTriNeighbor(pts, min_corner, voxel_size):
    """
    Args:
        pts B x N x 3
        min_corner 3
        voxel_size float
    """
    x = (pts - min_corner) / voxel_size
    # B x N x 8 x 3  B x N x 8 x 1
    neighbors_idxs, weights = trilinear_weight(x)
    neighbors = neighbors_idxs * voxel_size + min_corner
    return neighbors, weights

def create_meshgrid(D,H,W):
    Zs,Ys,Xs = torch.meshgrid(torch.arange(D), torch.arange(H),  torch.arange(W))
    return torch.stack([Xs,Ys,Zs], dim=-1)

@torch.no_grad()
def pruning_v2(model, nodes_flag, tile_center, tile_size, voxel_size, density=4):
    model.eval()

    device = nodes_flag.device

    D,H,W = nodes_flag[1:-1,1:-1,1:-1].shape 

    assert D % density == 0 and H % density == 0 and W % density == 0

    D = D // density
    H = H // density
    W = W // density

    # D x H x W x 3 
    regular_indices = create_meshgrid(D, H, W)

    # 减去 dilate  的 voxel_size 
    min_corner = tile_center - tile_size / 2.0 - voxel_size

    # D x H x W x 3
    regular_o = (regular_indices.float() * density + 0.5) * voxel_size + min_corner

    step = voxel_size 
    total_samples = []
    for x in range(density):
        for y in range(density):
            for z in range(density):
                offset = torch.tensor([x,y,z],dtype=torch.float32)
                total_samples.append(regular_o + offset*step)

    # density**3 x D x H x W x 3
    regular_sample = torch.stack(total_samples, dim=0)

    regular_sample = regular_sample.to(device)
    regular_sample = regular_sample.reshape(-1,3)

    _, regular_sigma = model(regular_sample)

    regular_sigma = regular_sigma.reshape(density**3,D,H,W)

    # D x H x W
    regular_sigma, _ = torch.max(regular_sigma, dim=0)

    regular_sigma = regular_sigma.repeat_interleave(density,dim=0)
    regular_sigma = regular_sigma.repeat_interleave(density,dim=1)
    regular_sigma = regular_sigma.repeat_interleave(density,dim=2)

    nodes_flag = torch.ones_like(nodes_flag) 
    temp = torch.ones_like(regular_sigma)

    temp[torch.where(torch.exp(-regular_sigma) > 0.5)] = -1

    nodes_flag[1:-1,1:-1,1:-1] = temp

    dilate_boundary(nodes_flag, nodes_flag.shape[0])

    model.train()

    return nodes_flag

@torch.no_grad()
def pruning(model, nodes_flag, tile_center, tile_size, voxel_size, density=2):

    model.eval()

    D,H,W = nodes_flag.shape 

    # D x H x W x 3 
    regular_indices = create_meshgrid(D, H, W)
    min_corner = tile_center - tile_size / 2.0 - voxel_size


    # D x H x W x 3
    regular_o = regular_indices.float() * voxel_size + min_corner

    # D x H x W x 3 
    # regular_sample = (regular_indices.float() + 0.5) * voxel_size + min_corner


    step = voxel_size / (density - 1.)
    total_samples = []
    for x in range(density):
        for y in range(density):
            for z in range(density):
                offset = torch.tensor([x,y,z],dtype=torch.float32)
                total_samples.append(regular_o + offset*step)
    
    # N x D x H x W x 3
    regular_sample = torch.stack(total_samples, dim=0)

    regular_sample = regular_sample.to(nodes_flag.device)
    regular_sample = regular_sample.reshape(-1,3)

    regular_sigma = []
    batchSize = 4096
    for i in range(0,regular_sample.shape[0], batchSize):
        batch = regular_sample[i:i+batchSize]
        _, sigma = model(batch)
        regular_sigma.append(sigma)
    
    regular_sigma = torch.cat(regular_sigma, dim=0)
    regular_sigma = regular_sigma.reshape(density**3,D,H,W)

    regular_sigma,_ = torch.max(regular_sigma, dim=0)

    nodes_flag = torch.ones_like(nodes_flag) 

    nodes_flag[torch.where(torch.exp(-regular_sigma) > 0.5)] = -1

    dilate_boundary(nodes_flag, D)
    # nodes_flag[torch.where(torch.exp(-regular_sigma) > 0.5)] = False

    model.train()

    return nodes_flag


@torch.no_grad()
def compute_voxels(model, nodes_flag, num_voxel, tile_center, tile_size):

    model.eval()

    device = nodes_flag.device

    dilate_num_voxel = num_voxel + 2
    voxel_size = tile_size / num_voxel
    min_corner = tile_center - tile_size / 2 - voxel_size
    min_corner = torch.from_numpy(min_corner).to(device).float()

    voxels = torch.zeros((dilate_num_voxel, dilate_num_voxel, dilate_num_voxel,4), device=device)
    non_empty = torch.where(nodes_flag != -1)

    regular_indices = create_meshgrid(dilate_num_voxel,dilate_num_voxel,dilate_num_voxel).to(device)
    regular_sample = (regular_indices.float() + 0.5) * voxel_size + min_corner

    regular_sample = regular_sample[non_empty]
    regular_rgb, regular_sigma = model(regular_sample.reshape(-1,3))
    voxels[non_empty] = torch.cat([regular_rgb, regular_sigma], dim=-1)
    voxels = voxels[None,...].permute(0,4,1,2,3) # 1 x 4 x D x H x W

    model.train()

    return voxels 


def cal_psnr(I1,I2):
    mse = torch.mean((I1-I2)**2)
    if mse < 1e-10:
        return 100
    return 10 * float(torch.log10(255.0**2/mse))

@torch.no_grad()
def important_reflection_sample(mirror_depth, nsamples, epsilon):
    device = mirror_depth.device
    near = mirror_depth - epsilon
    far = mirror_depth + epsilon
    step = torch.arange(nsamples, device=device).float() / (nsamples - 1.)
    z_vals = near + step * step 
    return z_vals

@torch.no_grad()
def inverse_z_sample(near, far, nsamples):
    device = near.device 
    inv_near = 1. / near.squeeze(-1) # B 
    inv_far = 1. / far # 1 
    inv_bound = inv_far - inv_near # B 

    step = torch.arange(nsamples,device=device).float() / (nsamples - 1.) # Nsamples

    z_vals = 1. / (step[None,:] * inv_bound[:,None] + inv_near[:,None])
    return z_vals

@torch.no_grad()
def reflection_sampling(rays_o, rays_d, node_flags, voxels, num_voxel, center, size, nsamples, far, thresh):
    device = rays_o.device
    z_vals = torch.full((rays_o.shape[0], nsamples), -1, dtype=torch.float32, device=device)
    Sample_reflection(rays_o, rays_d, node_flags.permute(2,1,0).contiguous(), 
                      voxels.permute(3,2,1,0).contiguous(),*center, size, far, num_voxel, thresh, z_vals)
    return z_vals



@torch.no_grad()
def sparse_sampling(rays_o, rays_d, node_flags, num_voxel, center, size, nsamples):
    """
        node_flags D x H x W
    """
    device = rays_o.device
    z_vals = torch.full((rays_o.shape[0], nsamples), -1, dtype=torch.float32, device=device)
    dists = torch.full((rays_o.shape[0], nsamples), -1, dtype=torch.float32, device=device)
    Sample_sparse(rays_o, rays_d, node_flags.permute(2,1,0).contiguous(), *center, size, num_voxel, z_vals, dists)
    return z_vals, dists

@torch.no_grad()
def uniform_sampling(rays_o, rays_d, center, size, nsamples):
    device = rays_o.device
    z_vals = torch.full((rays_o.shape[0], nsamples), -1, dtype=torch.float32, device=device)
    Sample_uniform(rays_o, rays_d, *center, size, z_vals)
    return z_vals


@torch.no_grad()
def sampling_background(rays_o, rays_d, bgdepth, tile_center, tile_size, nsamples, sample_range):
    device = rays_o.device
    z_vals = torch.full((rays_o.shape[0], nsamples), -1, dtype=torch.float32, device=device)
    Sample_bg(rays_o, rays_d, bgdepth, tile_center, tile_size, sample_range, z_vals)
    return z_vals


def volume_rendering(rgb, sigma, dists, init_T = 1.):
    """
    rgb B x Nsamples x 3 
    sigma B x Nsamples x 1
    dists B x Nsamples x 1 
    """
    B = rgb.shape[0]
    device = rgb.device
    alpha = 1. - torch.exp(-sigma*dists) # B x Nsamples x 1 
    # B x Nsamples x 1
    transparency = torch.cumprod(torch.cat([torch.ones(B,1,1, device=device)*init_T, 1-alpha+1e-10], 1), 1)[:,:-1,:]
    weights = alpha * transparency
    # B x 3 
    out = torch.sum(weights * rgb, dim=1)
    return out, transparency, alpha

def volume_rendering_v3(rgb, sigma, mat, dists, init_T = 1.):
    """
    rgb B x Nsamples x 3 
    sigma B x Nsamples x 1
    dists B x Nsamples x 1 
    """
    B = rgb.shape[0]
    device = rgb.device
    alpha = 1. - torch.exp(-sigma*dists) # B x Nsamples x 1 
    # B x Nsamples x 1
    transparency = torch.cumprod(torch.cat([torch.ones(B,1,1, device=device)*init_T, 1-alpha+1e-10], 1), 1)[:,:-1,:]
    weights = alpha * transparency
    # B x 3 
    out = torch.sum(weights * rgb, dim=1)
    out_mat = torch.sum(weights * mat, dim=1)
    return out, out_mat, transparency, alpha

def volume_rendering_v2(rgb, sigma, zvals, dists, init_T = 1.):
    """
    rgb B x Nsamples x 3 
    sigma B x Nsamples x 1
    dists B x Nsamples x 1 
    zvals B x Nsamples x 1 
    """
    B = rgb.shape[0]
    device = rgb.device
    alpha = 1. - torch.exp(-sigma*dists) # B x Nsamples x 1 
    # B x Nsamples x 1
    transparency = torch.cumprod(torch.cat([torch.ones(B,1,1, device=device)*init_T, 1-alpha+1e-10], 1), 1)[:,:-1,:]
    weights = alpha * transparency
    # B x 3 
    color = torch.sum(weights * rgb, dim=1)
    depth = torch.sum(weights * zvals, dim=1)
    return color, depth, transparency, alpha


def load_ignore(path, num_camera):
    ignore = []
    try:
        f = open(path, 'r')
    except:
        pass
    else:
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.strip().split(' ')
            if len(line) == 1:
                ignore += [int(line[0])]
            elif len(line) == 2:
                ignore += list(np.arange(int(line[0]), int(line[1])))
    out = np.zeros(num_camera, dtype=np.int32)
    for item in ignore:
        out[item] = 1
    return out 

def refine_visImg(imgIdx, tileCenter, H, W, Ks, C2Ws, ignore_path, dis_thresh=16, bounding=50, need=150):

    ignore = []
    try:
        f = open(ignore_path, 'r')
    except:
        pass
    else:
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.strip().split(' ')
            if len(line) == 1:
                ignore += [int(line[0])]
            elif len(line) == 2:
                ignore += list(np.arange(int(line[0]), int(line[1])))
    imgIdx = [item for item in imgIdx if item not in ignore]

    print(f"delete ignore, remain img num: {len(imgIdx)}")

    if len(imgIdx) < need:
        return np.array(imgIdx) 

    sims = []
    for idx in imgIdx:
        camera_center = C2Ws[idx,:,-1]
        vec = tileCenter - camera_center
        sim = np.dot(vec, np.array([0,0,1]))
        sims.append(sim)

    imgIdx = np.array(imgIdx)[np.argsort(np.array(sims))] 

    angle_order = []

    add_idx = [0]

    step = 1
    count = 1 
    while True:
        for i in range(1,step,2):
            add_idx += [int(i / step * (len(imgIdx)-1))]
            count += 1
            if count >= len(imgIdx):
                break
        if count >= len(imgIdx):
            break
        step = step * 2  

    for idx in add_idx:
        angle_order.append(imgIdx[idx])

    new_imgIdx = []
    for idx in angle_order:
        camera_center = C2Ws[idx,:,-1]
        vec = tileCenter - camera_center
        dis = np.linalg.norm(vec)
        pixel = Ks[idx] @ C2Ws[idx,:3,:3].transpose() @ (tileCenter - camera_center)
        pixel = pixel[:2] / (pixel[2] + 1e-8)
        if pixel[0] < bounding or pixel[0] >= W - bounding:
            continue 
        if pixel[1] < bounding or pixel[1] >= H - bounding:
            continue 
        
        if dis > dis_thresh:
            continue 
        new_imgIdx.append(idx)

    new_imgIdx = new_imgIdx[:need]

    if len(new_imgIdx) == 0:
        return np.array(imgIdx)


    return np.array(new_imgIdx)


def rotate_x(theta):
    R = np.array([[1, 0, 0], 
                      [0, np.cos(theta), -np.sin(theta)], 
                      [0, np.sin(theta), np.cos(theta)]], 
                      dtype=np.float32)
    return R 

def rotate_y(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)], 
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]], 
                      dtype=np.float32)
    return R 

def rotate_z(theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0], 
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]], 
                      dtype=np.float32)
    return R 

def get_360degeree_path(focus_center, distance, up, num=10):

    def form_C2W(z_axis):
        x_axis = np.cross(up, z_axis) # 3
        y_axis = np.cross(z_axis, x_axis) # 3 
        C = focus_center - z_axis * distance # 3 
        return np.stack([x_axis,y_axis,z_axis, C], axis= -1)
    
    start = np.array([0,-np.sin(np.pi/6.),np.cos(np.pi/6.)])
    C2Ws = []
    step = 2 / num 
    for theta in np.arange(0, 2, step):
        Ry = rotate_y(theta*np.pi)
        # Rz = rotate_z(45 * np.pi)
        # z_axis = Rz @ Ry @ start
        z_axis = Ry @ start
        C2W = form_C2W(z_axis)
        C2Ws.append(C2W)
    return C2Ws

# Hierarchical sampling (section 5.2)
@torch.no_grad()
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def get_trainVoxels(overlapVoxels, tileIdx, voxels, IndexMap, SparseToDense, num_voxel, overlap):
    """

    Args:
        tileIdx ([type]): [description]
        voxels ([type]): [description]
        SparseToDense ([type]): [description]
        num_voxel ([type]): [description]
        dilate_ratio (float, optional): [description]. Defaults to 0.25.
    """
    get_idx = lambda a,b,c: a * (num_voxel ** 2) + b * num_voxel + c 

    num_voxel_dilate = overlap
    num_voxel_train = num_voxel + num_voxel_dilate * 2

    initVoxel = np.zeros((num_voxel*3, num_voxel*3, num_voxel*3, 4))
    initVoxel[num_voxel:-num_voxel,
              num_voxel:-num_voxel,
              num_voxel:-num_voxel] = voxels[tileIdx].copy()

    
    overlapVoxels[tileIdx] = initVoxel[num_voxel-num_voxel_dilate:2*num_voxel+num_voxel_dilate,
                                       num_voxel-num_voxel_dilate:2*num_voxel+num_voxel_dilate,
                                       num_voxel-num_voxel_dilate:2*num_voxel+num_voxel_dilate]


def get_overlapVoxels(voxels, IndexMap, SparseToDense, num_voxel, overlap=1):
    overlapVoxels = np.zeros((voxels.shape[0], num_voxel+overlap*2,
                            num_voxel+overlap*2,num_voxel+overlap*2, 4), dtype=np.float32)
    
    for tileIdx in tqdm(range(len(voxels))):
        get_trainVoxels(overlapVoxels, tileIdx, voxels, IndexMap, SparseToDense, 
                                                num_voxel, overlap)

    return overlapVoxels




"""Bezier, a module for creating Bezier curves.
Version 1.1, from < BezierCurveFunction-v1.ipynb > on 2019-05-02
"""

class Bezier():
    def TwoPoints(t, P1, P2):
        """
        Returns a point between P1 and P2, parametised by t.
        INPUTS:
            t     float/int; a parameterisation.
            P1    numpy array; a point.
            P2    numpy array; a point.
        OUTPUTS:
            Q1    numpy array; a point.
        """

        if not isinstance(P1, np.ndarray) or not isinstance(P2, np.ndarray):
            raise TypeError('Points must be an instance of the numpy.ndarray!')
        if not isinstance(t, (int, float)):
            raise TypeError('Parameter t must be an int or float!')

        Q1 = (1 - t) * P1 + t * P2
        return Q1

    def Points(t, points):
        """
        Returns a list of points interpolated by the Bezier process
        INPUTS:
            t            float/int; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            newpoints    list of numpy arrays; points.
        """
        newpoints = []
        #print("points =", points, "\n")
        for i1 in range(0, len(points) - 1):
            #print("i1 =", i1)
            #print("points[i1] =", points[i1])

            newpoints += [Bezier.TwoPoints(t, points[i1], points[i1 + 1])]
            #print("newpoints  =", newpoints, "\n")
        return newpoints

    def Point(t, points):
        """
        Returns a point interpolated by the Bezier process
        INPUTS:
            t            float/int; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            newpoint     numpy array; a point.
        """
        newpoints = points
        #print("newpoints = ", newpoints)
        while len(newpoints) > 1:
            newpoints = Bezier.Points(t, newpoints)
            #print("newpoints in loop = ", newpoints)

        #print("newpoints = ", newpoints)
        #print("newpoints[0] = ", newpoints[0])
        return newpoints[0]

    def Curve(t_values, points):
        """
        Returns a point interpolated by the Bezier process
        INPUTS:
            t_values     list of floats/ints; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            curve        list of numpy arrays; points.
        """

        if not hasattr(t_values, '__iter__'):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if len(t_values) < 1:
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if not isinstance(t_values[0], (int, float)):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")

        curve = np.array([[0.0] * len(points[0])])
        for t in t_values:
            #print("curve                  \n", curve)
            #print("Bezier.Point(t, points) \n", Bezier.Point(t, points))

            curve = np.append(curve, [Bezier.Point(t, points)], axis=0)

            #print("curve after            \n", curve, "\n--- --- --- --- --- --- ")
        curve = np.delete(curve, 0, 0)
        #print("curve final            \n", curve, "\n--- --- --- --- --- --- ")
        return curve


class PairRandomCrop(object):
    """
    for (lr, hr) pair random crop
    """

    def __init__(self, lr_shape, lr_crop_shape, scale):
        self.lr_h, self.lr_w = lr_shape  
        self.hr_h, self.hr_w = [scale * x for x in lr_shape] 
        self.lr_crop_h, self.lr_crop_w = lr_crop_shape # 64 x 64
        self.hr_crop_h, self.hr_crop_w = self.lr_crop_h * scale, self.lr_crop_w * scale # 128 x 128
        self.lr_h_crop_start = random.randint(
            0, self.lr_h - self.lr_crop_h - 1)
        self.lr_w_crop_start = random.randint(
            0, self.lr_w - self.lr_crop_w - 1)
        self.hr_h_crop_start, self.hr_w_crop_start = self.lr_h_crop_start * \
            scale, self.lr_w_crop_start * scale

    def crop_with_hr_params(self, inputs):
        """
        inputs's shape: (N)HWC
        """
        outputs = inputs[
            ..., 
            self.hr_h_crop_start:self.hr_h_crop_start+self.hr_crop_h,
            self.hr_w_crop_start:self.hr_w_crop_start+self.hr_crop_w,
            :
        ]
        return outputs

    def crop_with_lr_params(self, inputs):
        """
        inputs's shape: (N)hwC
        """
        outputs = inputs[
            ...,
            self.lr_h_crop_start:self.lr_h_crop_start+self.lr_crop_h,
            self.lr_w_crop_start:self.lr_w_crop_start+self.lr_crop_w,
            :
        ]
        return outputs
    
class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)
        
        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw
        
    def __call__(self, img, scale=1):
        if len(img.shape) == 3:
            return img[self.h1 * scale: self.h2 * scale, self.w1 * scale: self.w2 * scale, :]
        else:
            return img[self.h1 * scale: self.h2 * scale, self.w1 * scale: self.w2 * scale, ...]


def get_grid(data, height, width, normalize=False):
    grid = compute_grid(data, height, width)
    grid = np.frombuffer(grid, dtype=np.float32).reshape((height,width,2))
    if normalize:
        grid = grid / [width-1, height-1] * 2 - 1
    return grid 

def warp_C(K_src, K_dst, E_src, E_dst, D_dst, height, width, normalize=False):
    data = np.concatenate((K_src, E_src[:3,:3].reshape((-1,)), E_src[:3,3:4].reshape((-1,)),
                           K_dst, E_dst[:3,:3].reshape((-1,)), E_dst[:3,3:4].reshape((-1,)),
                           D_dst.reshape((-1,)))).astype(np.float32)

    grid = compute_grid(data.tostring(), height, width)
    grid = np.frombuffer(grid, dtype=np.float32).reshape((height,width,2))
    if normalize:
        grid = grid / [width-1, height-1] * 2 - 1
    return grid


def update_depth_by_change_coor(depth_src, cam_src, cam_dst):

    height,width = depth_src.shape[:2]
    focal, px, py = cam_src[:3]

    E_src = np.zeros((4,4))
    E_src[:3,:3] = np.array(cam_src[3:12]).reshape((3,3))
    E_src[:3,3:4] = np.array(cam_src[12:15]).reshape((3,1))
    E_src[3:4,3:4] = 1.0

    E_dst = np.zeros((4,4))
    E_dst[:3,:3] = np.array(cam_dst[3:12]).reshape((3,3))
    E_dst[:3,3:4] = np.array(cam_dst[12:15]).reshape((3,1))
    E_dst[3:4,3:4] = 1.0

    transform_maxtrix = E_dst @ np.linalg.inv(E_src) # 4 x 4

    X,Y = np.meshgrid(np.arange(width),np.arange(height)) # H x W
    X = (X - px)/focal # HW1
    Y = (Y - py)/focal # HW1
    X = depth_src * X[..., None]
    Y = depth_src * Y[..., None]
    # H x W x 4
    src_cam_grad = np.concatenate([X, Y, 
                                   depth_src * np.ones_like(X), 
                                   np.ones_like(X)], axis=-1)
    src_cam_grad = src_cam_grad.reshape((height*width, 4)).transpose()
    dst_cam_grid = transform_maxtrix @ src_cam_grad # 4 x HW

    new_depth = dst_cam_grid[2,...].reshape((height, width))
    return new_depth[...,None]