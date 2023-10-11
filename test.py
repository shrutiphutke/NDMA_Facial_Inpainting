from __future__ import print_function
import argparse
import os
import cv2

import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import get_test_set

from utils import is_image_file, load_img, save_img
torch.backends.cudnn.benchmark = True
# Testing settings
parser = argparse.ArgumentParser(description='NDMAL-testing-code')
parser.add_argument('--dataset', default='./dataset/', required=False, help='facades')
parser.add_argument('--save_path', default='results', required=False, help='facades')
parser.add_argument('--checkpoints_path', default='checkpoints/', required=False, help='facades')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--cuda', action='store_false', help='use cuda')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.save_path+'/'):
        os.makedirs(opt.save_path+'/')


device = torch.device("cuda:0" if opt.cuda else "cpu")

G_path = opt.checkpoints_path+"netG_model.pth"
my_net = torch.load(G_path).to(device)  

test_set = get_test_set(opt.dataset)
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)


for iteration_test, batch in enumerate(testing_data_loader,1):
    input1, mask_test, target, image_filename = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3]

    prediction, prediction2 = my_net(input1, mask_test)
    prediction_img2 = prediction2.detach().squeeze(0).cpu()    
    save_img(prediction_img2, "./{}/{}".format(opt.save_path+'/',image_filename[0]))
 