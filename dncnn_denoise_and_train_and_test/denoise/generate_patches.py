import argparse
import glob
from PIL import Image
import PIL
import random
import numpy as np
import os
from utils import *

# the pixel value range is '0-255'(uint8 ) of training data

# macro
DATA_AUG_TIMES = 1  # transform a sample to a different sample for DATA_AUG_TIMES times

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./data/train/train_c', help='dir of data_c')
parser.add_argument('--src_dir_n', dest='src_dir_n', default='./data/train/train_n', help='dir of data_n')
parser.add_argument('--save_dir', dest='save_dir', default='./data', help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=50, help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=15, help='stride')
parser.add_argument('--step', dest='step', type=int, default=0, help='step')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=128, help='batch size')
# check output arguments
parser.add_argument('--from_file', dest='from_file', default="./data/img_clean_pats.npy", help='get pic_c from file')
parser.add_argument('--from_file_n', dest='from_file_n', default="./data/img_noise_pats.npy", help='get pic_n from file')
parser.add_argument('--num_pic', dest='num_pic', type=int, default=10, help='number of pic to pick')
args = parser.parse_args()


def generate_patches(isDebug=False):
    global DATA_AUG_TIMES
    count = 0
    data=np.load('/home/cpssface/hs/flowers_classification_1/denoise/tf_in_all_pats.npy')
    im=data[:,0,:,:,:].reshape(350,256,256,1)  
    im=(im+1)*128
    no_im=data[:,1,:,:,:].reshape(350,256,256,1) 
    no_im=(no_im+1)*128


    print ("number of training data %d" % data.shape[0])
    
    scales = [1, 0.9, 0.8, 0.7]
    
    # calculate the number of patches
    for i in range(data.shape[0]):
        img = im[i,:,:,:].reshape(256,256,1)

        
        img_n = no_im[i,:,:,:].reshape(256,256,1)
        
        for s in range(len(scales)):
            newsize = (int(img.shape[0] * scales[s]), int(img.shape[1] * scales[s]))
            img_s =np.resize(img,newsize)  
            img_s_n=np.resize(img_n,newsize) 

            # do not change the original img
            im_h, im_w = img_s.shape
            
            for x in range(0 + args.step, (im_h - args.pat_size), args.stride):
                for y in range(0 + args.step, (im_w - args.pat_size), args.stride):
                    count += 1
    origin_patch_num = count * DATA_AUG_TIMES
    
    if origin_patch_num % args.bat_size != 0:
        numPatches = (origin_patch_num / args.bat_size + 1) * args.bat_size
    else:
        numPatches = origin_patch_num
    print ("total patches = %d , batch size = %d, total batches = %d" % \
          (numPatches, args.bat_size, numPatches / args.bat_size))
    
    # data matrix 4-D
    inputs = np.zeros((int(numPatches),2, args.pat_size, args.pat_size, 1))
    count = 0
    # generate patches
    for i in range(data.shape[0]):
        img = im[i,:,:,:].reshape(256,256)
        
        img_n = no_im[i,:,:,:].reshape(256,256)
        

        for s in range(len(scales)):
            newsize = (int(img.shape[0] * scales[s]), int(img.shape[1] * scales[s]))
            img_s =np.resize(img,newsize)  
            img_s_n=np.resize(img_n,newsize) 
            

            
            # print newsize

            img_s = np.reshape(img_s,
                               (img_s.shape[0], img_s.shape[1], 1))  # extend one dimension

            img_s_n = np.reshape(img_s_n,
                                         (img_s_n.shape[0], img_s_n.shape[1], 1))
            
            for j in range(DATA_AUG_TIMES):
                im_h, im_w, _ = img_s.shape
                for x in range(0 + args.step, im_h - args.pat_size, args.stride):
                    for y in range(0 + args.step, im_w - args.pat_size, args.stride):
                        a=random.randint(0, 7)
                        inputs[count, 0,:, :, :] = data_augmentation(img_s[x:x + args.pat_size, y:y + args.pat_size, :], \
                                                                   a)
                        inputs[count, 1, :, :, :] = data_augmentation(img_s_n[x:x + args.pat_size, y:y + args.pat_size, :], \
                                                                   a)

                        
                        count += 1

 
            
    # pad the batch
    if count < numPatches:
        to_pad = int(numPatches - count)
        inputs[-to_pad:,:, :, :, :] = inputs[:to_pad, :,:, :, :]

    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    np.save(os.path.join(args.save_dir, "img_clean_and_noise_pats"), inputs)
    print ("size of inputs tensor = " + str(inputs.shape))
    


if __name__ == '__main__':
    generate_patches()
