from models import *
from utils import *
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.datasets as Dataset
import time
import os
import tqdm


data_root = '/media/bingchen/wander/stream_cycle_gan/data/renaissance/'

ndf = 64
ngf = 64
nz = 100
lr = 2e-4
betas = (0.5, 0.99)
manualSeed = 1234

batch_size = 64
max_iteration = 200000
lr_decay_start = max_iteration // 2

shuffle = True
dataloader_workers = 8
use_cuda = True	


torch.backends.cudnn.benchmark = True
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

#dataset = Dataset.ImageFolder(root=data_root, transform=trans_maker(128)) 
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=dataloader_workers)

loss_bce = nn.BCELoss()
loss_nll = nn.NLLLoss()
loss_mse = nn.MSELoss()
loss_l1 = nn.L1Loss()

dataset = Dataset.ImageFolder(root=data_root, transform=trans_maker(128)) 
dataloader = iter(DataLoader(
		dataset, batch_size,
		sampler=InfiniteSamplerWrapper(dataset),
		num_workers=dataloader_workers, pin_memory=True)
	)


save_folder = './'

log_interval = 100
save_image_interval = 200
save_model_interval = 1000




def save_image(netG, fixed_data, saved_image_folder, trial_name, itx):
	netG.eval()
	with torch.no_grad():
		#g_img = torch.cat([fixed_data, netG(fixed_data)], dim=0)
		
		g_img = netG(fixed_data)[0]
		#vutils.save_image( ( g_img + 1 ) * 0.5 ,  saved_image_folder+'/%s_iter_%d.jpg'%(trial_name, itx) )			
		vutils.save_image(  g_img ,  saved_image_folder+'/%s_iter_%d.jpg'%(trial_name, itx) )	
	netG.train()

def train(net, opt, device, trial_name, start_point=0):
	saved_image_folder, saved_model_folder = make_folders(save_folder, trial_name)

	fixed_data = next(dataloader)[0].to(device)

	avg_rec = avg_kl = 0
	for n_iter in tqdm.tqdm(range(start_point, max_iteration + 1)):

		if n_iter >= lr_decay_start:
			decay_lr(opt, max_iteration, lr_decay_start, lr)

		if n_iter % save_model_interval==0:
			save_model(net, multi_gpu, device, saved_model_folder, trial_name, n_iter)

		if n_iter % save_image_interval==0:
			save_image(net, fixed_data, saved_image_folder, trial_name, n_iter)
			
		## 1. prepare data
		real = next(dataloader)[0].to(device)

		net.zero_grad()
		#fake, feat_mu, feat_sigma, ca_mu, ca_sigma = net(real)
		fake, feat_mu, feat_sigma = net(real)

		#rec_loss = loss_l1(fake, real.detach())
		rec_loss = F.binary_cross_entropy(fake, real, reduction='sum')
		kl_loss = -0.5 * torch.sum(1 + feat_sigma - feat_mu.pow(2) - feat_sigma.exp()) 
		#	- 0.5 * torch.sum(1 + ca_sigma - ca_mu.pow(2) - ca_sigma.exp())

		total_loss = rec_loss + 3 * kl_loss

		total_loss.backward()
		optimizer.step()

		avg_rec += rec_loss.item()
		avg_kl += kl_loss.item()
		if n_iter % log_interval == 0:
			print( " recons_loss: %.5f    kl_loss: %.5f "%(avg_rec/log_interval, avg_kl/log_interval) )
			avg_rec = avg_kl = 0

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
multi_gpu = False
device = torch.device("cuda:1") if use_cuda else torch.device("cpu")

start_point = 0
checkpoint = None

trial_name = 'vae_da_attn_trial_1'

#net = VAE(nfc=64, nz=nz).to(device)
net = VAE_ATTN()
init_weights('orthogonal')(net)

#checkpoint = './train_results/vae_no_attn_trial_1/models/vae_no_attn_trial_1_iter_4000.pth'
#net.load_state_dict(torch.load(checkpoint))

net.to(device)

if use_cuda and multi_gpu:
	net = nn.DataParallel(net)

optimizer = optim.Adam(net.parameters(), lr=lr*10)#, betas=betas)

train(net, optimizer, device, trial_name)