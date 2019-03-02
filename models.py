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
			nn.Linear(ngf*16, ngf*4*8*8), nn.BatchNorm1d(ngf*4*8*8), nn.ReLU(),
			UnFlatten(8),
			nn.Conv2d(ngf*4, ngf*4, 3, 1, 1), nn.BatchNorm2d(ngf*4), nn.ReLU(),
			nn.Conv2d(ngf*4, ngf*4, 3, 1, 1), nn.BatchNorm2d(ngf*4), nn.ReLU(),
			nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1), nn.BatchNorm2d(ngf*2), nn.ReLU(),
			nn.ConvTranspose2d(ngf*2, ngf*1, 4, 2, 1), nn.BatchNorm2d(ngf*1), nn.ReLU(),
			nn.ConvTranspose2d(ngf*1, nc, 4, 2, 1), nn.Tanh())

	def r_sampler(self, z):
		code = self.e(z.view(z.size(0), -1))
		loc = code[:, :self.r_dim]
		scale = F.softplus(code[:, self.r_dim:]) + 1e-5
		scale_tri = torch.bmm( scale.view(-1, self.r_dim, 1), scale.view(-1, 1, self.r_dim) )
		return MultivariateNormal(loc=loc, scale_tril=scale_tri)

	def generate(self, r):
		return self.g(r)

	def forward(self, z):
		sampler = self.r_sampler(z)
		r = sampler.sample()
		img = self.generate(r)
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
			nn.Conv2d(nc, ndf, 4, 2, 1), nn.LeakyReLU(0.2),
			nn.Conv2d(ndf, ndf*2, 4, 2, 1), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2),
			nn.Conv2d(ndf*2, ndf*4, 4, 2, 1), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2),
			nn.Conv2d(ndf*4, ndf*4, 3, 1, 1), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2),
			nn.Conv2d(ndf*4, ndf*4, 3, 1, 1), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2),
			nn.Conv2d(ndf*4, ndf*16, 8, 1, 0), nn.BatchNorm2d(ndf*16), nn.LeakyReLU(0.2),
			Flatten()
		)

		self.q = nn.Sequential(
			nn.Linear(ndf*16, z_dim), nn.BatchNorm1d(z_dim), nn.ReLU(),
			nn.Linear(z_dim, z_dim))
		self.d = nn.Linear(ndf*16, 1)

	def discriminate(self, x):
		feat = self.feature(x)
		return self.d(feat)

	def posterior(self, x):
		feat = self.feature(x)
		return self.q(feat)

	def forward(self, x):
		feat = self.feature(x)
		d = self.d(feat)
		q = self.q(feat)
		return d, q