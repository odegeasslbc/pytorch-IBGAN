import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter, init
from torch.nn.utils import spectral_norm


img = torch.Tensor(3, 3, 128, 128).normal_(0, 1)
z = torch.Tensor(3, 128).normal_(0, 1)



class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
	def forward(self, input, size=1024):
		return input.view(input.size(0), size, 1, 1)


class VAE_simple(nn.Module):
	def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
		super(VAE_simple, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=4, stride=2),
			nn.ReLU(),
			Flatten()
		)
		
		self.fc1 = nn.Linear(h_dim, z_dim)
		self.fc2 = nn.Linear(h_dim, z_dim)
		self.fc3 = nn.Linear(z_dim, h_dim)
		
		self.decoder = nn.Sequential(
			UnFlatten(),
			nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
			nn.ReLU(),
			nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
			nn.Sigmoid(),
		)
		
	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		# return torch.normal(mu, std)
		esp = torch.randn(*mu.size()).to(mu.device)
		z = mu + std * esp
		return z
	
	def bottleneck(self, h):
		mu, logvar = self.fc1(h), self.fc2(h)
		z = self.reparameterize(mu, logvar)
		return z, mu, logvar

	def encode(self, x):
		h = self.encoder(x)
		z, mu, logvar = self.bottleneck(h)
		return z, mu, logvar

	def decode(self, z):
		z = self.fc3(z)
		z = self.decoder(z)
		return z

	def forward(self, x):
		z, mu, logvar = self.encode(x)
		z = self.decode(z)
		return z, mu, logvar
###################################################################
#######   attention modules
###################################################################
class Channel_Self_Attn(nn.Module):
	""" Self attention Layer"""
	def __init__(self, c, h, w, sigmoid=True):
		super(Channel_Self_Attn,self).__init__()
		self.c = c
		self.h = h
		self.w = w

		self.sigmoid = sigmoid

		self.query_m = nn.Parameter( torch.Tensor(h*w, c//8).normal_(0, 1) )
		self.key_m = nn.Parameter( torch.Tensor(h*w, c//8).normal_(0, 1) )
		self.value_conv = nn.Conv2d(in_channels = c , out_channels = c , kernel_size= 1)
		self.gamma = nn.Parameter(torch.zeros(1))

		self.softmax  = nn.Softmax(dim=-2)  # it has to be the -2 dim to do softmax, because
											# we later will transpose the value_feature_map
											# from C x N into N x C
	
	def forward(self,x):
		"""
			inputs :
				x : input feature maps( B X C X W X H)
			returns :
				out : channel-wise self attention value + input feature 
				attention: B X N X N (N is Width*Height)
		"""
		m_batchsize, C, width, height = x.size()

		proj_query  = torch.bmm( x.view(m_batchsize, C, -1), self.query_m.expand(m_batchsize, self.h*self.w, self.c//8) )  # B x C x C'
		proj_key = torch.bmm( x.view(m_batchsize, C, -1), self.key_m.expand(m_batchsize, self.h*self.w, self.c//8) ).transpose(1,2) # B X C' x C
		energy =  torch.bmm( proj_query, proj_key )  # B x C x C 
		if self.sigmoid:
			energy = torch.sigmoid(energy)
		attention = self.softmax( energy ) # B X (C) X (C)

		proj_value = self.value_conv(x).view(m_batchsize, C, width*height).transpose(1,2) # B x N x C

		out = torch.bmm(proj_value, attention).transpose(1,2)  # B x C x N
		out = out.view(m_batchsize, C, width, height)
		'''
		out = self.gamma*out + x
		return out
		'''
		return out
#csa = Channel_Self_Attn(32,8,8)

class Decoupling_Attn(nn.Module):
	""" Passive Attention Layer
		learns a fixed attention mask for every channel (c) of feature map (hxw)
	"""
	def __init__(self, c, h, w, softmax=True):
		super(Decoupling_Attn, self).__init__()
		self.c = c
		self.h = h
		self.w = w 
		self.attention = nn.Parameter(torch.Tensor(c, h * w).normal_(0, 1))
		self.softmax = softmax
		self.gamma = nn.Parameter(torch.zeros(1))

	def forward(self, input):
		#assert input.shape[1:] == torch.Size([self.c, self.h, self.w]), "Passive_Attention layer: wrong input shape"
		
		if self.softmax:
			return input + self.gamma * input * torch.softmax(self.attention, dim=1).view(self.c, self.h, self.w).expand_as(input)
		else:
			return input + self.gamma * input * self.attention.view(self.c, self.h, self.w).expand_as(input)
		
		#if self.softmax:
		#	return torch.softmax(self.attention, dim=1).view(self.c, self.h, self.w)#.expand_as(input)
		#else:
		#	return self.attention.view(self.c, self.h, self.w)#.expand_as(input)

class Self_Attn(nn.Module):
	""" Self attention Layer"""
	def __init__(self,in_dim):
		super(Self_Attn,self).__init__()
		self.chanel_in = in_dim
		
		self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
		self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
		self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
		self.gamma = nn.Parameter(torch.zeros(1))

		self.softmax  = nn.Softmax(dim=-1) #
	def forward(self,x):
		"""
			inputs :
				x : input feature maps( B X C X W X H)
			returns :
				out : self attention value + input feature 
				attention: B X N X N (N is Width*Height)
		"""
		m_batchsize,C,width ,height = x.size()
		proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
		proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
		energy =  torch.bmm(proj_query,proj_key) # transpose check
		attention = self.softmax(energy) # BX (N) X (N) 
		proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

		out = torch.bmm(proj_value,attention.permute(0,2,1) )
		out = out.view(m_batchsize,C,width,height)
		
		out = self.gamma*out + x
		return out#,attention


def _upsample(x):
	h, w = x.size()[2:]
	return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear', align_corners=True)


class VAE(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self, nfc=64, nc=3, nz=128):
		super(VAE, self).__init__()
		self.nfc = nfc

		self.encode_conv_128 = nn.Sequential(
			spectral_norm( nn.Conv2d(nc, nfc, 3, 1, 1, bias=False) ),
			nn.LeakyReLU(0.1, inplace=True),

			spectral_norm( nn.Conv2d(nfc, nfc*2, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*2),
			nn.LeakyReLU(0.1, inplace=True),
			
			spectral_norm( nn.Conv2d(nfc*2, nfc*4, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*4),
			nn.LeakyReLU(0.1, inplace=True)
			)

		self.encode_conv_32 = nn.Sequential(
			spectral_norm( nn.Conv2d(nfc*4, nfc*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*8),
			nn.LeakyReLU(0.2, inplace=False),

			spectral_norm( nn.Conv2d(nfc*8, nfc*16, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*16),
			nn.LeakyReLU(0.2, inplace=False),

			spectral_norm( nn.Conv2d(nfc*16, nfc*16, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*16),
			nn.LeakyReLU(0.2, inplace=False),

			spectral_norm( nn.Conv2d(nfc*16, nfc*16, 4, 1, 0, bias=False)),
			)

		self.encode_fc_feat = nn.Linear(nfc*16, nfc*8)
		self.encode_fc_feat_sigma = nn.Sequential(\
			nn.Dropout(), nn.ReLU(), nn.Linear(nfc*8, nz))
		self.encode_fc_feat_mu = nn.Sequential(\
			nn.Dropout(), nn.ReLU(), nn.Linear(nfc*8, nz))


		self.decode_fc_feat = nn.Sequential(
			nn.Linear(nz, nfc*4), nn.Dropout(), nn.ReLU(), nn.Linear(nfc*4, nfc*8*2*2))

		self.decode_conv_32 = nn.Sequential(
			spectral_norm( nn.ConvTranspose2d(nfc*8, nfc*16, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*16),
			nn.ReLU(),

			spectral_norm( nn.ConvTranspose2d(nfc*16, nfc*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*8),
			nn.ReLU(),

			spectral_norm( nn.ConvTranspose2d(nfc*8, nfc*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*8),
			nn.ReLU(),

			spectral_norm( nn.ConvTranspose2d(nfc*8, nfc*4, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*4),
			nn.ReLU(),
		)


		self.decode_conv_128 = nn.Sequential(
			spectral_norm( nn.ConvTranspose2d(nfc*4, nfc*2, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*2),
			nn.ReLU(),

			spectral_norm( nn.ConvTranspose2d(nfc*2, nfc, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc),
			nn.ReLU(),

			spectral_norm( nn.Conv2d(nfc, nc, 3, 1, 1, bias=False)),
			nn.Sigmoid()
		)

	def encode(self, x):
		feat = self.encode_conv_128(x)
		feat = self.encode_conv_32(feat).view(x.size(0), -1)
		
		feat = self.encode_fc_feat(feat)
		feat_mu = self.encode_fc_feat_mu(feat)
		feat_sigma = self.encode_fc_feat_sigma(feat)
		
		return feat_mu, feat_sigma

	def reparametrize(self, feat_mu, feat_sigma):
		feat_std = feat_sigma.mul(0.5).exp_()
		feat_eps = torch.FloatTensor(feat_sigma.size()).normal_().to(feat_mu.device)
	
		return feat_eps.mul(feat_std).add_(feat_mu)

	def decode(self, z_feat):
		feat = self.decode_conv_32( self.decode_fc_feat( z_feat.view(z_feat.size(0), -1) ).view(z_feat.size(0), 512, 2, 2 ) )
		return self.decode_conv_128(feat)

	def forward(self, input):
		feat_mu, feat_sigma = self.encode(input)
		feat_z = self.reparametrize(feat_mu, feat_sigma)
		result = self.decode(feat_z)
		return result, feat_mu, feat_sigma


class VAE_ATTN(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self, nfc=64, nc=3, nz=128):
		super(VAE_ATTN, self).__init__()
		self.nfc = nfc

		self.encode_conv_128 = nn.Sequential(
			spectral_norm( nn.Conv2d(nc, nfc, 3, 1, 1, bias=False) ),
			nn.LeakyReLU(0.1, inplace=True),

			spectral_norm( nn.Conv2d(nfc, nfc*2, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*2),
			nn.LeakyReLU(0.1, inplace=True),
			
			Decoupling_Attn(nfc*2, 64, 64),

			spectral_norm( nn.Conv2d(nfc*2, nfc*4, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*4),
			nn.LeakyReLU(0.1, inplace=True)
			)

		self.decouple_attn = Decoupling_Attn(nfc*4, 32, 32)
		#self.channel_attn = Channel_Self_Attn(nfc*4, 32, 32)
		#self.gamma_da = nn.Parameter(torch.zeros(1))
		#self.gamma_ca = nn.Parameter(torch.zeros(1))

		self.encode_conv_32 = nn.Sequential(
			spectral_norm( nn.Conv2d(nfc*4, nfc*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*8),
			nn.LeakyReLU(0.2, inplace=False),

			Decoupling_Attn(nfc*8, 16, 16),

			spectral_norm( nn.Conv2d(nfc*8, nfc*16, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*16),
			nn.LeakyReLU(0.2, inplace=False),

			Decoupling_Attn(nfc*16, 8, 8),

			spectral_norm( nn.Conv2d(nfc*16, nfc*16, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*16),
			nn.LeakyReLU(0.2, inplace=False),

			spectral_norm( nn.Conv2d(nfc*16, nfc*16, 4, 1, 0, bias=False)),
			)

		self.encode_fc_feat = nn.Linear(nfc*16, nfc*8)
		self.encode_fc_feat_sigma = nn.Sequential(\
			nn.Dropout(), nn.ReLU(), nn.Linear(nfc*8, nz))
		self.encode_fc_feat_mu = nn.Sequential(\
			nn.Dropout(), nn.ReLU(), nn.Linear(nfc*8, nz))
		'''
		self.encode_ca = nn.Sequential(
			nn.AvgPool2d(2),
			spectral_norm( nn.Conv2d(nfc*4, nfc*4, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*4),
			nn.LeakyReLU(0.2, inplace=False),
			nn.AvgPool2d(2),
			spectral_norm( nn.Conv2d(nfc*4, nfc*2, 4, 1, 0, bias=False)),
			nn.BatchNorm2d(nfc*2),
			nn.LeakyReLU(0.2, inplace=False))
		self.encode_fc_ca_sigma = nn.Sequential(\
			nn.Dropout(), nn.ReLU(), nn.Linear(nfc*2, nz))
		self.encode_fc_ca_mu = nn.Sequential(\
			nn.Dropout(), nn.ReLU(), nn.Linear(nfc*2, nz))
		'''
		self.decode_fc_feat = nn.Sequential(
			nn.Linear(nz, nfc*4), nn.Dropout(), nn.ReLU(), nn.Linear(nfc*4, nfc*8*2*2))

		self.decode_conv_32 = nn.Sequential(
			spectral_norm( nn.ConvTranspose2d(nfc*8, nfc*16, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*16),
			nn.ReLU(),

			spectral_norm( nn.ConvTranspose2d(nfc*16, nfc*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*8),
			nn.ReLU(),

			Decoupling_Attn(nfc*8, 8, 8),

			spectral_norm( nn.ConvTranspose2d(nfc*8, nfc*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*8),
			nn.ReLU(),

			Decoupling_Attn(nfc*8, 16, 16),

			spectral_norm( nn.ConvTranspose2d(nfc*8, nfc*4, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*4),
			nn.ReLU(),
		)
		'''
		self.decode_ca = nn.Sequential(
			spectral_norm( nn.ConvTranspose2d(nz, nfc*2, 4, 1, 0, bias=False)),
			nn.BatchNorm2d(nfc*2),
			nn.ReLU(),

			spectral_norm( nn.ConvTranspose2d(nfc*2, nfc*4, 4, 4, 0, bias=False)),
			nn.BatchNorm2d(nfc*4),
			nn.ReLU(),
			
			spectral_norm( nn.ConvTranspose2d(nfc*4, nfc*4, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*4),
			nn.ReLU())
		'''
		self.decode_conv_128 = nn.Sequential(
			spectral_norm( nn.ConvTranspose2d(nfc*4, nfc*2, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*2),
			nn.ReLU(),

			Decoupling_Attn(nfc*2, 64, 64),
			
			spectral_norm( nn.ConvTranspose2d(nfc*2, nfc, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc),
			nn.ReLU(),

			spectral_norm( nn.Conv2d(nfc, nc, 3, 1, 1, bias=False)),
			nn.Sigmoid()
		)
		'''
		self.encode_fc_da = nn.Linear(32*32, 512)
		self.encode_fc_da_sigma = nn.Sequential(\
			nn.Dropout(), nn.ReLU(), nn.Linear(512, nz))
		self.encode_fc_da_mu = nn.Sequential(\
			nn.Dropout(), nn.ReLU(), nn.Linear(512, nz))
		'''

		
	def encode(self, x):
		feat = self.encode_conv_128(x)
		feat = self.decouple_attn(feat)
		#da = self.decouple_attn(feat)  # size: c x w x h
		#feat = self.gamma_da * da.expand_as(feat) * feat + feat
		#ca = self.channel_attn(feat)   # size: b x c x w x h
		#feat = self.gamma_ca * ca * feat + feat
		feat = self.encode_conv_32(feat).view(x.size(0), -1)
		
		feat = self.encode_fc_feat(feat)
		feat_mu = self.encode_fc_feat_mu(feat)
		feat_sigma = self.encode_fc_feat_sigma(feat)
		
		#ca_h = self.encode_ca(ca).view(x.size(0), -1)
		#ca_mu = self.encode_fc_ca_mu(ca_h)
		#ca_sigma = self.encode_fc_ca_sigma(ca_h)

		return feat_mu, feat_sigma#, ca_mu, ca_sigma

	def reparametrize(self, feat_mu, feat_sigma):#, ca_mu, ca_sigma):
		feat_std = feat_sigma.mul(0.5).exp_()
		feat_eps = torch.FloatTensor(feat_sigma.size()).normal_().to(feat_mu.device)
		
		#ca_std = ca_sigma.mul(0.5).exp_()
		#ca_eps = torch.FloatTensor(feat_sigma.size()).normal_().to(ca_mu.device)

		return feat_eps.mul(feat_std).add_(feat_mu)#, ca_eps.mul(ca_std).add_(ca_mu)

	def decode(self, z_feat):#, z_ca):
		feat = self.decode_conv_32( self.decode_fc_feat( z_feat.view(z_feat.size(0), -1) ).view(z_feat.size(0), 512, 2, 2 ) )
		#ca = self.decode_ca( z_ca.view(z_ca.size(0), -1, 1, 1) )
		#da = self.decouple_attn(feat)
		#feat = self.gamma_da * da.expand_as(feat) * feat + feat
		#feat = self.gamma_ca * ca * feat + feat
		feat = self.decouple_attn(feat)
		return self.decode_conv_128(feat)

	def forward(self, input):
		#feat_mu, feat_sigma, ca_mu, ca_sigma = self.encode(input)
		#feat_z, ca_z = self.reparametrize(feat_mu, feat_sigma, ca_mu, ca_sigma)
		feat_mu, feat_sigma = self.encode(input)
		feat_z = self.reparametrize(feat_mu, feat_sigma)
		result = self.decode(feat_z)
		return result, feat_mu, feat_sigma#, ca_mu, ca_sigma


class AE(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self, nfc=64, nc=3, nz=128):
		super(AE, self).__init__()
		self.nfc = nfc

		self.encode_conv_128 = nn.Sequential(
			spectral_norm( nn.Conv2d(nc, nfc, 3, 1, 1, bias=False) ),
			nn.LeakyReLU(0.1, inplace=True),

			spectral_norm( nn.Conv2d(nfc, nfc*2, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*2),
			nn.LeakyReLU(0.1, inplace=True),
			
			spectral_norm( nn.Conv2d(nfc*2, nfc*4, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*4),
			nn.LeakyReLU(0.1, inplace=True)
			)

		self.encode_conv_32 = nn.Sequential(
			spectral_norm( nn.Conv2d(nfc*4, nfc*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*8),
			nn.LeakyReLU(0.2, inplace=False),

			spectral_norm( nn.Conv2d(nfc*8, nfc*16, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*16),
			nn.LeakyReLU(0.2, inplace=False),

			spectral_norm( nn.Conv2d(nfc*16, nfc*16, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*16),
			nn.LeakyReLU(0.2, inplace=False),

			spectral_norm( nn.Conv2d(nfc*16, nfc*16, 4, 1, 0, bias=False)),
			)

		self.decode_conv_32 = nn.Sequential(
			spectral_norm( nn.ConvTranspose2d(nfc*16, nfc*16, 4, 1, 0, bias=False)),
			nn.BatchNorm2d(nfc*16),
			nn.ReLU(),

			spectral_norm( nn.ConvTranspose2d(nfc*16, nfc*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*8),
			nn.ReLU(),

			spectral_norm( nn.ConvTranspose2d(nfc*8, nfc*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*8),
			nn.ReLU(),

			spectral_norm( nn.ConvTranspose2d(nfc*8, nfc*4, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*4),
			nn.ReLU(),
		)

		self.decode_conv_128 = nn.Sequential(
			spectral_norm( nn.ConvTranspose2d(nfc*4, nfc*2, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*2),
			nn.ReLU(),

			spectral_norm( nn.ConvTranspose2d(nfc*2, nfc, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc),
			nn.ReLU(),

			spectral_norm( nn.Conv2d(nfc, nc, 3, 1, 1, bias=False)),
			nn.Tanh()
		)

	def encode(self, x):
		feat = self.encode_conv_128(x)
		feat = self.encode_conv_32(feat)
		return feat

	def decode(self, feat):
		feat = self.decode_conv_32( feat )
		return self.decode_conv_128(feat)

	def forward(self, input):
		feat = self.encode(input)
		result = self.decode(feat)
		return result

class AE_ATTN(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self, nfc=64, nc=3, nz=128):
		super(AE_ATTN, self).__init__()
		self.nfc = nfc

		self.encode_conv_128 = nn.Sequential(
			spectral_norm( nn.Conv2d(nc, nfc, 3, 1, 1, bias=False) ),
			nn.LeakyReLU(0.1, inplace=True),

			spectral_norm( nn.Conv2d(nfc, nfc*2, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*2),
			nn.LeakyReLU(0.1, inplace=True),
			
			spectral_norm( nn.Conv2d(nfc*2, nfc*4, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*4),
			nn.LeakyReLU(0.1, inplace=True)
			)

		self.decouple_attn = Decoupling_Attn(nfc*4, 32, 32)
		self.channel_attn = Channel_Self_Attn(nfc*4, 32, 32)
		self.gamma_da = nn.Parameter(torch.zeros(1))
		self.gamma_ca = nn.Parameter(torch.zeros(1))

		self.encode_conv_32 = nn.Sequential(
			spectral_norm( nn.Conv2d(nfc*4, nfc*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*8),
			nn.LeakyReLU(0.2, inplace=False),

			spectral_norm( nn.Conv2d(nfc*8, nfc*16, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*16),
			nn.LeakyReLU(0.2, inplace=False),

			spectral_norm( nn.Conv2d(nfc*16, nfc*16, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*16),
			nn.LeakyReLU(0.2, inplace=False),

			spectral_norm( nn.Conv2d(nfc*16, nfc*16, 4, 1, 0, bias=False)),
			)

		self.decode_conv_32 = nn.Sequential(
			spectral_norm( nn.ConvTranspose2d(nfc*16, nfc*16, 4, 1, 0, bias=False)),
			nn.BatchNorm2d(nfc*16),
			nn.ReLU(),

			spectral_norm( nn.ConvTranspose2d(nfc*16, nfc*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*8),
			nn.ReLU(),

			spectral_norm( nn.ConvTranspose2d(nfc*8, nfc*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*8),
			nn.ReLU(),

			spectral_norm( nn.ConvTranspose2d(nfc*8, nfc*4, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*4),
			nn.ReLU(),
		)

		self.decode_conv_128 = nn.Sequential(
			spectral_norm( nn.ConvTranspose2d(nfc*4, nfc*2, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc*2),
			nn.ReLU(),

			spectral_norm( nn.ConvTranspose2d(nfc*2, nfc, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(nfc),
			nn.ReLU(),

			spectral_norm( nn.Conv2d(nfc, nc, 3, 1, 1, bias=False)),
			nn.Tanh()
		)

	def encode(self, x):
		feat = self.encode_conv_128(x)
		da = self.decouple_attn(feat)  # size: c x w x h
		feat = self.gamma_da * da.expand_as(feat) * feat + feat
		ca = self.channel_attn(feat)   # size: b x c x w x h
		feat = self.gamma_ca * ca * feat + feat
		feat = self.encode_conv_32(feat)

		return feat

	def decode(self, feat):
		feat = self.decode_conv_32( feat )
		da = self.decouple_attn(feat)
		feat = self.gamma_da * da.expand_as(feat) * feat + feat
		ca = self.channel_attn(feat)
		feat = self.gamma_ca * ca * feat + feat

		return self.decode_conv_128(feat)

	def forward(self, input):
		feat = self.encode(input)
		result = self.decode(feat)
		return result




class SAGenerator(nn.Module):
	"""docstring for _Generator"""
	def __init__(self, ngf=64, nz=100, nc=3):
		super(SAGenerator, self).__init__()
		self.main = nn.Sequential(
			spectral_norm( nn.ConvTranspose2d(nz, ngf*16, 4, 1, 0, bias=False)),
			nn.BatchNorm2d(ngf*16),
			nn.ReLU(True),

			spectral_norm( nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ngf*8),
			nn.ReLU(True),

			spectral_norm( nn.ConvTranspose2d(ngf*8, ngf*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ngf*8),
			nn.ReLU(True),

			Self_Attn(ngf*8),

			spectral_norm( nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ngf*4),
			nn.ReLU(True),

			Self_Attn(ngf*4),

			spectral_norm( nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ngf*2),
			nn.ReLU(True),

			spectral_norm( nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),

			spectral_norm( nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)),
			nn.Tanh()
			)
		
	def forward(self, input):
		return self.main(input)


class SADiscriminator(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self, ndf=32, nc=3):
		super(SADiscriminator, self).__init__()
		self.ndf = ndf

		self.main = nn.Sequential(
			spectral_norm( nn.Conv2d(nc, ndf, 4, 2, 1, bias=False) ),
			nn.LeakyReLU(0.2, inplace=True),

			spectral_norm( nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf*2),
			nn.LeakyReLU(0.2, inplace=True),

			spectral_norm( nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf*4),
			nn.LeakyReLU(0.2, inplace=True),

			Self_Attn(ndf*4),

			spectral_norm( nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf*8),
			nn.LeakyReLU(0.2, inplace=True),

			Self_Attn(ndf*8),

			spectral_norm( nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf*16),
			nn.LeakyReLU(0.2, inplace=True),

			spectral_norm( nn.Conv2d(ndf*16, ndf*16, 4, 2, 1, bias=False)),
			nn.BatchNorm2d(ndf*16),
			nn.LeakyReLU(0.2, inplace=True),

			spectral_norm( nn.Conv2d(ndf*16, 1, 4, 1, 0, bias=False)),
			)

	def forward(self, input):
		return self.main(input).view(-1)


