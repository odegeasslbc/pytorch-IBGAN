import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import subprocess as sp
from PIL import Image
import time
from torch.nn import Parameter, init
import torch.utils.data as data



def init_weights(init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    return init_func


def decay_lr(opt, max_iter, start_iter, initial_lr):
    """Decay learning rate linearly till 0."""
    coeff = -initial_lr / (max_iter - start_iter)
    for pg in opt.param_groups:
        pg['lr'] += coeff


def _rescale(img):
    return img * 2.0 - 1.0

def _noise_adder(img):
    return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1/128.0) + img

def trans_maker(size=256):
	trans = transforms.Compose([ 
					transforms.Resize((size+36,size+36)),
					transforms.RandomHorizontalFlip(),
					transforms.RandomCrop((size, size)),
					transforms.ToTensor(),
					#_rescale, _noise_adder
					])
	return trans

# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L5-L15
def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L18-L26
class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def make_folders(save_folder, trial_name):
	saved_model_folder = os.path.join(save_folder, 'train_results/%s/models'%trial_name)
	saved_image_folder = os.path.join(save_folder, 'train_results/%s/images'%trial_name)
	folders = [os.path.join(save_folder, 'train_results'), os.path.join(save_folder, 'train_results/%s'%trial_name), 
	os.path.join(save_folder, 'train_results/%s/images'%trial_name), os.path.join(save_folder, 'train_results/%s/models'%trial_name)]

	for folder in folders:
		if not os.path.exists(folder):
			os.mkdir(folder)
	return saved_image_folder, saved_model_folder


def save_model(net, multi_gpu, device, saved_model_folder, trial_name, iter):
	print('saving models ...')
	if multi_gpu:
		torch.save( net.module.cpu().state_dict(), \
				saved_model_folder+'/%s_iter_%d.pth'%(trial_name, iter) )
	else:
		torch.save( net.cpu().state_dict(), \
				saved_model_folder+'/%s_iter_%d.pth'%(trial_name, iter) )
	print('saving models done')
	net.to(device)
