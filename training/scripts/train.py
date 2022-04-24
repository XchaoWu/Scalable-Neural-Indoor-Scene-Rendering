import numpy as np 
import utils
from tile.tileScene import TileScene
import random 
import torch 
import sys,os 
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-c","--cfg", help="config file path", type=str, required=True)
parser.add_argument("-g","--gpu", help="gpu idx", type=int, default=0)
parser.add_argument("-ts","--tfs", help="path of tiles idx for trainig", 
                           type=str, default="tileIdxs.txt")
parser.add_argument("-t","--tileIdx", help="tileIdx for training, just for single tile training",
                            type=int, default=-1)
parser.add_argument("-p","--parallel", help="parallel training between groups", type=bool, default=False)
parser.add_argument("-r", "--refine", help="second iteration of training", type=bool, default=False )


args = parser.parse_args()


cfg = utils.parse_yaml(args.cfg)

seed = cfg.SEED
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

TS = TileScene(cfg, args.refine, args.cfg)

if args.parallel == False:
    if args.tileIdx != -1:
        prefix = f"single-tile-{args.tileIdx}"
        tileIdx_list = [args.tileIdx]
    elif args.tfs == 'all':
        prefix = f"all"
        IndexMap = TS.cfg.IndexMap
        tileIdx_list = [item for item in IndexMap if item != -1]
    else:
        prefix = f"{os.path.splitext(os.path.basename(os.path.split(args.tfs)[1]))[0]}"
        try:
            f = open(args.tfs, 'r')
        except:
            print("plz specify tileIdx")
            exit()  
        else:
            lines = f.readlines()
            tileIdx_list = []
            for item in lines:
                item = item.strip()
                if item == '#':
                    break
                tileIdx_list.append(int(item))
            f.close() 
    print("training tileIdxs", tileIdx_list)

    TS.train_tiles(tileIdx_list=tileIdx_list, gpu=args.gpu, prefix=prefix)
else:
    TS.parallel_train()