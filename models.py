import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter, init
from torch.nn.utils import spectral_norm
from torch.distributions.multivariate_normal import MultivariateNormal


class UnFlatten(nn.Module):
	def __init__(self, block_size):
		super(UnFlatten, self).__init__()
		self.block_size = block_size

	def forward(self, x):
		return x.view(x.size(0), -1, self.block_size, self.block_size)

class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)


class Generator_64(nn.Module):
	"""
		Generator_64 contains the model e for sampling r from z
		and the model g for generating image from sampled r
    """
	def __init__(self, ngf=64, z_dim=500, r_dim=15, nc=3):
		super(Generator_64, self).__init__()
		self.r_dim = r_dim
		self.e = nn.Sequential(
            nn.Linear(z_dim, ngf*2), nn.BatchNorm1d(ngf*2), nn.ReLU(),
            nn.Linear(ngf*2, ngf), nn.BatchNorm1d(ngf), nn.ReLU(),
            nn.Linear(ngf, r_dim*2))
		self.g = nn.Sequential(
			nn.Linear(r_dim, ngf*16), nn.BatchNorm1d(ngf*16), nn.ReLU(),
			nn.Linear(ngf*16, ngf*4*4*4), nn.BatchNorm1d(ngf*4*4*4), nn.ReLU(),
			UnFlatten(4),
			#nn.Conv2d(ngf*4, ngf*4, 3, 1, 1), nn.BatchNorm2d(ngf*4), nn.ReLU(),
			nn.ConvTranspose2d(ngf*4, ngf*4, 4, 2, 1), nn.BatchNorm2d(ngf*4), nn.ReLU(),
			nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1), nn.BatchNorm2d(ngf*2), nn.ReLU(),
			nn.ConvTranspose2d(ngf*2, ngf*1, 4, 2, 1), nn.BatchNorm2d(ngf*1), nn.ReLU(),
			nn.ConvTranspose2d(ngf*1, nc, 4, 2, 1), nn.Tanh())

	def r_sampler(self, z):
		code = self.e(z.view(z.size(0), -1))
		# old: using a torch distribution to sample r
		mu = code[:, :self.r_dim]
		var = F.softplus(code[:, self.r_dim:]) + 1e-5
		scale_tri = torch.diag_embed(var)
		return MultivariateNormal(loc=mu, scale_tril=scale_tri)
		# new: a easier reparameterization
		#mu = code[:, :self.r_dim]
		#logvar = code[:,self.r_dim:]
		#r = torch.randn(*mu.size()).to(mu.device)
		#r = mu + r * logvar.mul_(0.5).exp_()
		#return r, mu, logvar 

	def generate(self, r):
		return self.g(r)

	def forward(self, z):
		r_distribution = self.r_sampler(z)
		img = self.generate(r_distribution.sample())
		return img

class Discriminator_64(nn.Module):
	"""
		Discriminator_64 contains the model q for inferring z_hat
		and the model d for discriminating real/fake images
		d and q share the feature extracting modules
    """
	def __init__(self, ndf=64, z_dim=500, nc=3):
		super(Discriminator_64, self).__init__()
		
		self.feature = nn.Sequential(
			spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1)), nn.ReLU(),
			spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1)), nn.BatchNorm2d(ndf*2), nn.ReLU(),
			spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)), nn.BatchNorm2d(ndf*4), nn.ReLU(),
			#nn.Conv2d(ndf*4, ndf*4, 3, 1, 1), nn.BatchNorm2d(ndf*4), nn.ReLU(),
			#nn.Conv2d(ndf*4, ndf*4, 3, 1, 1), nn.BatchNorm2d(ndf*4), nn.ReLU(),
			spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)), nn.BatchNorm2d(ndf*8), nn.ReLU(),
			
		)

		self.q = nn.Sequential(
			Flatten(),
			nn.Linear(ndf*8*16, ndf*16), nn.BatchNorm1d(ndf*16), nn.ReLU(),
			nn.Linear(ndf*16, z_dim))
		self.d = spectral_norm(nn.Conv2d(ndf*8, 1, 4, 1, 0))

	def discriminate(self, x):
		feat = self.feature(x)
		return self.d(feat).view(-1)

	def posterior(self, x):
		feat = self.feature(x)
		return self.q(feat)

	def forward(self, x):
		feat = self.feature(x)
		d = self.d(feat)
		q = self.q(feat)
		return d.view(-1), q