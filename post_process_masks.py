from argparse import _StoreTrueAction
import torch
import PIL.Image as Image
from multiprocessing import Pool
import scipy.spatial as sp
import numpy as np

import argparse
import glob
import os.path as osp
import cv2
import os

def map_rgb(img,main_colors):
    
    tree=sp.KDTree(main_colors)
    for py in range(0, img.shape[0]):
        for px in range(0, img.shape[1]):
            color=img[py][px]
            _,result=tree.query(color)
            img[py][px]=torch.tensor(main_colors[result])
            
    return img

def map_binary(img):
    img=torch.tensor(img)
    return np.array(img.apply_(lambda x: 255 if x > 255/2 else 0))

def map_img(args,path):
    img = cv2.imread( path, cv2.IMREAD_UNCHANGED)

    if args.binary:
        img=map_binary(img)
    else:
        img=map_rgb(img, main_colors)

    cv2.imwrite(
        osp.join(args.output_path, osp.basename(path)),
        img
    )

main_colors=[]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--output_path')
    parser.add_argument('--binary', action=_StoreTrueAction)
    parser.add_argument('--threads', type=int, default=5)


    args = parser.parse_args()
    print(args)

    os.makedirs(args.output_path, exist_ok=True)
    files=glob.glob(osp.join(args.input_path, "*.png"))

    print(f'files detected: {len(files)}')

    def curried_map_img(path):
        map_img(args,path)
    with Pool(args.threads) as p:
        print(p.map(curried_map_img, files))