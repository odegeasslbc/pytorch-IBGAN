import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence

import torchvision.datasets as Dataset
from torchvision import utils as vutils

import tqdm
from itertools import chain
from os.path import join as pjoin
import numpy as np

from models import Generator_64, Discriminator_64
from utils import trans_maker, InfiniteSamplerWrapper, make_folders, save_model, save_image_from_z, save_image_from_r


BATCH_SIZE = 128
Z_DIM = 500
R_DIM = 15
NDF = 64
NGF = 64
MAX_ITERATION = 100000

LAMBDA_G = 1
BETA_KL = 0.3

LR_G = 5e-5
LR_E = 5e-5
LR_Q = 5e-5
LR_D = 5e-5

DATA_ROOT = "../img_align_celeba/"
DATALOADER_WORKERS = 8

SAVE_FOLDER = './'
TRIAL_NAME = 'IBGAN_celeba_64_trial_1'
LOG_INTERVAL = 100
SAVE_IMAGE_INTERVAL = 200
SAVE_MODEL_INTERVAL = MAX_ITERATION//50

CHECKPOINT = None #'/media/bingchen/wander/ibgan/train_results/trial_1/models/15000.pth'

CUDA = 0
MULTI_GPU = False

device = torch.device("cpu")
if CUDA > -1:
	device = torch.device("cuda:%d"%(CUDA))

dataset = Dataset.ImageFolder(root=DATA_ROOT, transform=trans_maker(64)) 
dataloader = iter(DataLoader(dataset, BATCH_SIZE, \
	sampler=InfiniteSamplerWrapper(dataset), num_workers=DATALOADER_WORKERS, pin_memory=True))

loss_bce = nn.BCELoss()
loss_mse = nn.MSELoss()

M_r = MultivariateNormal(loc=torch.zeros(R_DIM).to(device), scale_tril=torch.ones(R_DIM, R_DIM).to(device))

def KL_Loss(z):
	mu = z.mean()
	logvar = z.var().log()
	return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def train(netG, netD, opt_G, opt_D, opt_E):
	D_real = D_fake = D_z_kl = G_real = Z_recon = R_kl = 0
	fixed_z = torch.randn(64, Z_DIM).to(device)

	saved_image_folder, saved_model_folder = make_folders(SAVE_FOLDER, TRIAL_NAME)

	for n_iter in tqdm.tqdm(range(0, MAX_ITERATION+1)):

		if n_iter % SAVE_IMAGE_INTERVAL == 0:
			save_image_from_z(netG, fixed_z, pjoin(saved_image_folder, "z_%d.jpg"%n_iter))
			save_image_from_r(netG, R_DIM, pjoin(saved_image_folder, "r_%d.jpg"%n_iter))
		if n_iter % SAVE_MODEL_INTERVAL == 0:
			save_model(netG, netD, pjoin(saved_model_folder, "%d.pth"%n_iter))	
		
		### 0. prepare data
		real_image = next(dataloader)[0].to(device)

		z = torch.randn(BATCH_SIZE, Z_DIM).to(device)
		# e(r|z) as the likelihood of r given z
		r_sampler = netG.r_sampler(z)
		g_image = netG.generate(r_sampler.sample())

		### 1. Train Discriminator on real and generated data
		netD.zero_grad()
		pred_f = netD.discriminate(g_image.detach())
		pred_r, rec_z = netD(real_image)
		d_loss = loss_bce(torch.sigmoid(pred_r), torch.ones(pred_r.size()).to(device)) \
			+ loss_bce(torch.sigmoid(pred_f), torch.zeros(pred_f.size()).to(device))
		q_loss = KL_Loss(rec_z)
		#d_loss.backward()
		total_loss = d_loss + q_loss
		total_loss.backward()
		opt_D.step()

		# record the loss values
		D_real += torch.sigmoid(pred_r).mean().item()
		D_fake += torch.sigmoid(pred_f).mean().item()
		D_z_kl += q_loss.item()

		### 2. Train Generator
		netD.zero_grad()
		netG.zero_grad()
		# q(z|x) as the posterior of z given x
		pred_g, z_posterior = netD(g_image)
		# GAN loss for generator
		g_loss = LAMBDA_G * loss_bce(torch.sigmoid(pred_g), torch.ones(pred_g.size()).to(device))
		# reconstruction loss of z
		## TODO
		## question here: as stated in the paper-algorithm-1: this part should be a - log(q(z|x)) instead of mse
		recon_loss = loss_mse(z_posterior, z)
		# kl loss between e(r|z) || m(r) as a variational inference
		#kl_loss = BETA_KL * torch.distributions.kl.kl_divergence(r_likelihood, M_r).mean()
		kl_loss = BETA_KL * kl_divergence(r_sampler, M_r).mean()
		total_loss = g_loss + recon_loss + kl_loss
		total_loss.backward()
		opt_E.step()
		opt_G.step()

		# record the loss values
		G_real += torch.sigmoid(pred_g).mean().item()
		Z_recon += recon_loss.item()
		R_kl += kl_loss.item()

		if n_iter % LOG_INTERVAL == 0 and n_iter > 0:
			print("D(x): %.5f    D(G(z)): %.5f    D_kl: %.5f    G(z): %.5f    Z_rec: %.5f    R_kl: %.5f"%\
				(D_real/LOG_INTERVAL, D_fake/LOG_INTERVAL, D_z_kl/LOG_INTERVAL, G_real/LOG_INTERVAL, Z_recon/LOG_INTERVAL, R_kl/LOG_INTERVAL))
			D_real = D_fake = D_z_kl = G_real = Z_recon = R_kl = 0

if __name__ == "__main__":
	
	netG = Generator_64(ngf=NGF, z_dim=Z_DIM, r_dim=R_DIM)
	netD = Discriminator_64(ndf=NDF, z_dim=Z_DIM)

	if CHECKPOINT is not None:
		ck = torch.load(CHECKPOINT)
		netG.load_state_dict(ck['g'])
		netD.load_state_dict(ck['d'])

	netG.to(device)
	netD.to(device)
	
	opt_G = optim.RMSprop(netG.g.parameters(), lr=LR_G, momentum=0.9)
	opt_E = optim.RMSprop(netG.e.parameters(), lr=LR_E, momentum=0.9)
	#opt_Q = optim.RMSprop( chain(netD.feature.parameters(), netD.q.parameters()), lr=LR_Q, momentum=0.9)
	opt_D = optim.RMSprop( netD.parameters(), lr=LR_D, momentum=0.9)
	'''
	opt_G = optim.Adam(netG.g.parameters(), lr=LR_G, betas=(0.5, 0.99))
	opt_D = optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.99))
	opt_E = optim.Adam(netG.e.parameters(), lr=LR_E, betas=(0.5, 0.99))
	'''
	train(netG, netD, opt_G, opt_D, opt_E)