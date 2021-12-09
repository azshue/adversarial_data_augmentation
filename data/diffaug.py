import math
import torch
import torchvision
# from utils.transforms import rgb2hsv, hsv2rgb, rgb2yuv
from kornia.color import rgb_to_hsv, hsv_to_rgb, rgb_to_yuv



class DiffAugment(object):
	def __init__(self, eps):
		self.dims = 4
		self.channels = 3
		self.max = 1.0
		self.min = 0.0
		self.eps = eps
		self.dist = torch.distributions.normal.Normal(0, 1)
		sample = self.dist.sample(torch.Size([1, 3, 66, 200]))
		self.sample = sample.squeeze_(dim=-1).cuda()
		self.id_map = {
			0: "1", 1: "2",
			2: "R", 3: "G", 4: "B",
			5: "H", 6: "S", 7: "V",
		}

	def resample(self):
		sample = self.dist.sample(torch.Size([1, 3, 66, 200]))
		self.sample = sample.squeeze_(dim=-1).cuda()
		
	#1
	# https://github.com/tensorflow/addons/blob/v0.11.2/tensorflow_addons/image/filters.py#L226
	def blur(self, img, sigma, size=20):
		# sigma = torch.tensor(sigma)

		x = torch.arange(-size //2 + 1, size // 2 + 1)
		x = (x ** 2).type_as(sigma)
		x = torch.exp(-x / (2.0 * (sigma ** 2)))
		x = x / torch.sum(x)
		x1 = torch.reshape(x, (size, 1))
		x2 = torch.reshape(x, (1, size))

		gaussian_kernel = torch.matmul(x1, x2)
		channels = img.size()[1]
		gaussian_kernel = torch.stack((gaussian_kernel, gaussian_kernel, gaussian_kernel), dim=0)
		gaussian_kernel.unsqueeze_(dim=1).requires_grad_(True)

		paddings = ((size - 1) // 2, size - 1 - (size - 1) // 2,
					(size - 1) // 2, size - 1 - (size - 1) // 2)
		img = torch.nn.functional.pad(img, paddings)

		output = torch.nn.functional.conv2d(
			input=img,
			weight=gaussian_kernel,
			stride=1,
			groups=channels
		)
		return output
	#2
	def gaussian(self, img, magnitude, noise=None):
		if noise is None:
			noise = magnitude * self.sample
		else:
			noise = magnitude * noise
		img = img + noise
		img =  torch.clamp(img, 0., 1.)
		return img
	#R
	def color_R(self, img, magnitude):
		output = torch.stack((img[:, 0, ...], img[:, 1, ...], img[:, 2, ...] * (1 + magnitude)), dim=1)
		img = torch.clamp(output, 0., 1.)
		img = output
		return img
	#G
	def color_G(self, img, magnitude):
		output = torch.stack([img[:, 0, ...], img[:, 1, ...] * (1 + magnitude), img[:, 2, ...]], dim=1)
		img = torch.clamp(output, 0., 1.)
		return img
	#B
	def color_B(self, img, magnitude):
		output = torch.stack([img[:, 0, ...] * (1 + magnitude), img[:, 1, ...], img[:, 2, ...]], dim=1)
		img = torch.clamp(output, 0., 1.)
		return img
	#H
	def color_H(self, img, magnitude):
		hsv_img = rgb_to_hsv(img)
		output = torch.stack((hsv_img[:, 0, ...] * (1 + magnitude), hsv_img[:, 1, ...], hsv_img[:, 2, ...]), dim=1)
		output = torch.clamp(output, 0., 2.* math.pi)
		img = hsv_to_rgb(output)
		return img
	#S
	def color_S(self, img, magnitude):
		hsv_img = rgb_to_hsv(img)
		output = torch.stack((hsv_img[:, 0, ...], hsv_img[:, 1, ...] * (1 + magnitude), hsv_img[:, 2, ...]), dim=1) 
		output = torch.clamp(output, 0., 1.)
		img = hsv_to_rgb(output)
		return img
	#V
	def color_V(self, img, magnitude):
		hsv_img = rgb_to_hsv(img)
		output = torch.stack((hsv_img[:, 0, ...], hsv_img[:, 1, ...], hsv_img[:, 2, ...] * (1 + magnitude)), dim=1) 
		output =torch.clamp(output, 0., 1.)
		img = hsv_to_rgb(output)
		return img

	def single_aug(self, img, aug_id, delta, noise=None):
		if aug_id == '1': # gaussian blur
			aug_op = getattr(self, "blur") # change to Aug class
			param = delta + 1.
			param_min = 0.0
		elif aug_id == '2': # gaussian noise
			aug_op = getattr(self, "gaussian")
			param = delta
			param_min = -self.eps
		elif aug_id in ['R', 'G', 'B', 'H', 'S', 'V']: 
			aug_op = getattr(self, "color_" + aug_id)
			param = delta
			param_min = -self.eps
		elif aug_id == 'N':
			aug_op = None
			param_min = None
		else:
			print("augmentation is not defined: ", aug_id)

		if aug_op is not None:
			if noise is not None and aug_id == '2':
				img = aug_op(img, param, noise)
			else:
				img = aug_op(img, param)
		return img, param_min

	def __call__(self, img, aug_id, delta, noise=None, save=-1):
		assert(len(img.size()) == self.dims)
		assert(img.size()[1] == self.channels)
		assert(torch.max(img) <= self.max)
		assert(torch.min(img) >= self.min)

		if len(aug_id) == 1 and delta.size(0) == 1:
			### single aug
			img, param_min = self.single_aug(img, aug_id, delta)
			img = rgb_to_yuv(img)
			img = (img - 0.5) / 0.5
			return img, param_min
		else:
			### combined aug (used at test time)
			# delta: a random vector with 3 entris, each controls the magnitude of a single augmentation
			for i, id in enumerate(aug_id):
				img, _ = self.single_aug(img, self.id_map[id], delta[i], noise)
			if save >= 0:
				# save some example images
				save_grid = torchvision.utils.make_grid(img[:8], "./vis_comb.png")
				torchvision.utils.save_image(save_grid)
			img = rgb_to_yuv(img)
			img = (img - 0.5) / 0.5
			return img