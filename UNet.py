from math import ceil
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
import cv2
from image_processing import read_img


class UNet(nn.Module):
    """
    Implementation of UNet architecture
    """

    def __init__(self, n_class):
        super().__init__()

        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)


        #Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        #Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        
        #Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        #Output layer
        out = self.outconv(xd42)

        return out
    


def load_training_patches(  image_paths:str, mask_paths:str,
                            size: int = 512,patch_size: int = 256,
                            percent_of_images: float = 0.5,
                            only_green: bool = True ):
    
    """
    Create patches from given images
    
    :param image_paths: Path to data to train on
    :type image_paths: str
    :param mask_paths: Path to expert mask (ground truth)
    :type mask_paths: str
    :param size: Size of input images
    :type size: int
    :param patch_size: Size of a patch
    :type patch_size: int
    :param percent_of_images: Percentage amount of images to use in training
    :type percent_of_images: float
    :param only_green: DSwitch to decide if extract green channel
    :type only_green: bool
    """




    x_patches = []
    y_patches = []

    max_id = ceil(len(image_paths) * percent_of_images)

    for idx, (img_p, mask_p) in enumerate(zip(image_paths, mask_paths)):

        img = read_img(str(img_p))
        img = cv2.resize(img, (size, size))

        # extracting green channel
        if only_green:
            img = img[:, :, 1]

        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (size, size))

        h, w = img.shape[:2]

        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):

                patch_img = img[y:y + patch_size, x:x + patch_size]
                patch_mask = mask[y:y + patch_size, x:x + patch_size]

                if np.max(patch_img) > 15:

                    patch_img = patch_img.astype(np.float32) / 255.0
                    patch_mask = patch_mask.astype(np.float32) / 255.0


                    patch_img = np.expand_dims(patch_img, axis=0)


                    patch_mask = np.expand_dims(patch_mask, axis=0)

                    x_patches.append(patch_img)
                    y_patches.append(patch_mask)

        if idx >= max_id:
            break

    x_patches = np.array(x_patches, dtype=np.float32)
    y_patches = np.array(y_patches, dtype=np.float32)

    return x_patches, y_patches
