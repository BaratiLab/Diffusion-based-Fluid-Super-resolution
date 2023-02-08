import torch
import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def download_process_data(path="colab_demo"):
    os.makedirs(path, exist_ok=True)
    print("Downloading data")
    torch.hub.download_url_to_file('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/colab_examples/lsun_bedroom1.pth', os.path.join(path, 'lsun_bedroom1.pth'))
    torch.hub.download_url_to_file('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/colab_examples/lsun_bedroom2.pth', os.path.join(path, 'lsun_bedroom2.pth'))
    torch.hub.download_url_to_file('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/colab_examples/lsun_bedroom3.pth', os.path.join(path, 'lsun_bedroom3.pth'))
    torch.hub.download_url_to_file('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/colab_examples/lsun_edit.pth', os.path.join(path, 'lsun_edit.pth'))
    torch.hub.download_url_to_file('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/colab_examples/lsun_church.pth', os.path.join(path, 'lsun_church.pth'))
    print("Data downloaded")
    
def normalize_array(x):
    # x: input numpy array with shape (W, H)
    x_min = np.amin(x)
    x_max = np.amax(x)
    y = (x - x_min) / (x_max - x_min)
    return y, x_min, x_max

def unnormalize_array(y, x_min, x_max):
    return y * (x_max - x_min) + x_min

def data_blurring(fno_data_sample):
    # fno_data_sample: torch tensor, size: [64, 64]
    # return: blurred torch tensor, size: [64, 64]

    ds_size = 16
    resample_method = Image.NEAREST

    x_array, x_min, x_max = normalize_array(fno_data_sample.numpy())
    im = Image.fromarray((x_array*255).astype(np.uint8))
    im_ds = im.resize((ds_size, ds_size))
    im_us = im_ds.resize((im.width, im.height), resample=resample_method)
    x_array_blur = np.asarray(im_us)

    # inverse-transform to the original value range
    x_array_blur = x_array_blur.astype(np.float32)/255.0
    x_array_blur = unnormalize_array(x_array_blur, x_min, x_max)

    return torch.from_numpy(x_array_blur)
