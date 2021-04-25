#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys, yaml, time
import copy
import torch
import pickle, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from models.RITnet_v2 import DenseNet2D
from args import parse_args
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from helperfunctions import mypause, linVal
from pytorchtools import EarlyStopping, load_from_file
from utils import get_nparams, Logger, get_predictions, lossandaccuracy
from utils import getSeg_metrics, getPoint_metric, generateImageGrid, unnormPts
from utils import getAng_metric
from test import calc_acc
from bdcn_new import BDCN
from torchvision.utils import make_grid
import argparse
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

#%%
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Deactive file locking
embed_log = 5
EPS=1e-7
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)
def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return
def parse_args():
    parser = argparse.ArgumentParser(description='Train BDCN for different args')

    parser.add_argument('--curObj', type=str, help='select curriculum to train on', required=True)
    parser.add_argument('--config', type=str, default='hp.yaml',
                        help='Path to the config file.')
    parser.add_argument('--param-dir', type=str, default='params',
        help='the directory to store the params')
    parser.add_argument('--lr', dest='base_lr', type=float, default=1e-4,
        help='the base learning rate of model')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
        help='the momentum')
    parser.add_argument('-c', '--cuda', action='store_true', default=1,
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0, 1',
        help='the gpu id to train net')
    parser.add_argument('--weight-decay', type=float, default=0.0002,
        help='the weight_decay of net')
    parser.add_argument('-r', '--resume', type=str, default=0,
        help='whether resume from some, default is None')
    parser.add_argument('-p', '--pretrain', type=str, default=None,
        help='init net from pretrained model default is None')
    parser.add_argument('--maxiter', type=int, default=40000,
        help='max iters to train network, default is 40000')
    parser.add_argument('--iter-size', type=int, default=10,
        help='iter size equal to the batch size, default 10')
    parser.add_argument('--average-loss', type=int, default=50,
        help='smoothed loss, default is 50')
    parser.add_argument('-s', '--snapshots', type=int, default=1000,
        help='how many iters to store the params, default is 1000')
    parser.add_argument('--step-size', type=int, default=10000,
        help='the number of iters to decrease the learning rate, default is 10000')
    parser.add_argument('--display', type=int, default=20,
        help='how many iters display one time, default is 20')
    parser.add_argument('-b', '--balance', type=float, default=1.1,
        help='the parameter to balance the neg and pos, default is 1.1')
    parser.add_argument('-l', '--log', type=str, default='log.txt',
        help='the file to store log, default is log.txt')
    parser.add_argument('-k', type=int, default=1,
        help='the k-th split set of multicue')
    # batch_size
    parser.add_argument('--batch-size', type=int, default=1,
        help='batch size of one iteration, default 1')
    parser.add_argument('--crop-size', type=int, default=None,
        help='the size of image to crop, default not crop')
    parser.add_argument('--yita', type=float, default=None,
        help='the param to operate gt, default is data in the config file')
    parser.add_argument('--complete-pretrain', type=str, default=None,
        help='finetune on the complete_pretrain, default None')
    parser.add_argument('--side-weight', type=float, default=0.5,
        help='the loss weight of sideout, default 0.5')
    parser.add_argument('--fuse-weight', type=float, default=1.1,
        help='the loss weight of fuse, default 1.1')
    parser.add_argument('--gamma', type=float, default=0.1,
        help='the decay of learning rate, default 0.1')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    device=torch.device("cuda")

    # Open relevant train/test object
    f = open(os.path.join(os.getcwd(), 'baseline', 'cond_'+str(args.curObj)+'.pkl'), 'rb')

    # Get splits
    _, _, testObj = pickle.load(f)
    testObj.path2data = os.path.join('../../', 'Datasets', 'TEyeD-h5-Edges')
    # testObj.path2data = os.path.join('../../', 'Datasets', 'All/New')
    testObj.augFlag = False

    trainloader = DataLoader(testObj,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=8,
                             drop_last=True)

    # load model
    BDCN_network = BDCN()

    state_dict = torch.load('gen_00000016.pt')
    BDCN_network.load_state_dict(state_dict['a'])
    if args.cuda:
        BDCN_network = BDCN_network.cuda()
    BDCN_network.eval()

    # calc time for analyse
    start_time = time.time()

    # print edge
    start_edge = time.time()
    print('!!!Test ', args.batch_size)
    ans = []
    for bt, batchdata in enumerate(trainloader):
        if(bt > 62):break
        img, labels, spatialWeights, distMap, pupil_center, iris_center, elNorm, cond, imInfo = batchdata
        with torch.no_grad():
            a = torch.cat((img, img, img), dim=1).to(device).to(torch.float32)
            ans.append(a.shape)
            img_edge = BDCN_network(torch.cat((img, img, img), dim=1).to(device).to(torch.float32))[-1]

    print('test edge time : {:.3f}'.format(time.time() - start_edge))
    print('num:', len(ans))
