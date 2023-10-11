from os import listdir
from os.path import join
import random

from PIL import Image
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img

class DatasetFromFolder_Test(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder_Test, self).__init__()
        self.a_path = join(image_dir, "input/")
        self.mask_path = join(image_dir, "mask/")
        self.b_path = join(image_dir, "target/")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a = cv2.imread(join(self.a_path, self.image_filenames[index]))
        mask = cv2.imread(join(self.mask_path, self.image_filenames[index]))
        b = cv2.imread(join(self.b_path, self.image_filenames[index]))

        a1 = cv2.resize(a, (256, 256),  interpolation = cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (256, 256),  interpolation = cv2.INTER_CUBIC)
        b1 = cv2.resize(b, (256, 256),  interpolation = cv2.INTER_CUBIC)

        a1 = transforms.ToTensor()(a1)
        mask = transforms.ToTensor()(mask)
        b1 = transforms.ToTensor()(b1)
    
        a1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a1)
        mask = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(mask)
        b1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b1)

        return a1, mask, b1, self.image_filenames[index] 

    def __len__(self):
        return len(self.image_filenames)


