# Functions in this file are not used. Updated version uses the kronia package instead. 

import os
import glob
import cv2
import numpy as np

import torch

def rgb2hsv(img):
	out = torch.zeros_like(img)
	# v channel
	max_c, _ = torch.max(img, dim=1)
	out_v = max_c
	# s channel
	min_c, _ = torch.min(img, dim=1)
	delta = max_c - min_c
	out_s = delta / max_c
	out_s = torch.where(delta == 0.0, torch.tensor(0.0).cuda(), out_s)
	# h channel
	tmp_1 = (img[:, 1, ...] - img[:, 2, ...]) / delta[...]
	out_1 = torch.where(img[:, 0, ...] == max_c, tmp_1, torch.tensor(0.).cuda())
	tmp_2 = 2.0 + (img[:, 2, ...] - img[:, 0, ...]) / delta[...]
	out_2 = torch.where(img[:, 1, ...] == max_c, tmp_2, out_1)
	tmp_3 = 4.0 + (img[:, 0, ...] - img[:, 1, ...]) / delta[...]
	out_h = torch.where(img[:, 2, ...] == max_c, tmp_3, out_2)
	out_h = (out_h / 6.0) % 1.0
	out_h = torch.where(delta == 0.0, torch.tensor(0.0).cuda(), out_h)
	out = torch.stack([out_h, out_s, out_v], dim=1)
	out = torch.where(torch.isnan(x=out), torch.tensor(0.0).cuda(), out)
	return out

def hsv2rgb(img):
	hi = torch.floor(x=img[:, 0, ...] * 6)
	f = img[:, 0,...] * 6 - hi
	p = img[:, 2,...] * (1 - img[:, 1, ...])
	q = img[:, 2,...] * (1 - f * img[:, 1, ...])
	t = img[:, 2,...] * (1 - (1 - f) * img[:, 1, ...])
	v = img[:, 2,...]
	hi = torch.stack([hi, hi, hi], axis=1).type(torch.int32) % 6
	out_1 = torch.where(hi == 0, torch.stack([v, t, p], dim=1), torch.tensor(0.).cuda())
	out_2 = torch.where(hi == 1, torch.stack([q, v, p], dim=1), out_1)
	out_3 = torch.where(hi == 2, torch.stack([p, v, t], dim=1), out_2)
	out_4 = torch.where(hi == 3, torch.stack([p, q, v], dim=1), out_3)
	out_5 = torch.where(hi == 4, torch.stack([t, p, v], dim=1), out_4)
	out = torch.where(hi == 5, torch.stack([v, p, q], dim=1), out_5)
	return out

def rgb2yuv(img):
	return torch.stack((
		0.299 * img[:, 0, ...] + 0.587 * img[:, 1, ...] + 0.114 * img[:, 2, ...],
		-0.14714119 * img[:, 0, ...] + -0.28886916 * img[:, 1, ...] + 0.43601035 * img[:, 2, ...],
		0.61497538 * img[:, 0, ...] + -0.51496512 * img[:, 1, ...] + -0.10001026 * img[:, 2, ...],
	), dim=1)

