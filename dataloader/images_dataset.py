from torch.utils.data import Dataset
from PIL import Image
from imageio import imread
from util import data_utils
import numpy as np
import random
import clip
import random
import json
import os
import torch
import torchvision.transforms as transforms
from .mask_generator_256 import RandomMask
import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]

def natural_sort(items):
    items.sort(key=natural_keys)


class ImagesDataset_psp(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None, use_mask=False, return_name=False, datamax=None, hole_range=[0.5, 1], mask_root=None):
		self.source_paths = data_utils.make_dataset(source_root, datamax)
		self.target_paths = data_utils.make_dataset(target_root, datamax)
		if mask_root is not None:
			self.mask_paths = data_utils.make_dataset(mask_root, datamax)
		else:
			self.mask_paths = None
		natural_sort(self.source_paths[0])
		natural_sort(self.source_paths[1])
		natural_sort(self.target_paths[0])
		natural_sort(self.target_paths[1])
		if self.mask_paths is not None:
			natural_sort(self.mask_paths[0])
			natural_sort(self.mask_paths[1])
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts
		self.return_name = return_name
		self._use_mask = use_mask
		self.hole_range = hole_range
		self._len = len(self.source_paths[0])

	def __len__(self):
		return self._len

	def resize(self, img, height, width, centerCrop=True):
		imgh, imgw = img.shape[0:2]

		if centerCrop and imgh != imgw:
			# center crop
			side = np.minimum(imgh, imgw)
			j = (imgh - side) // 2
			i = (imgw - side) // 2
			img = img[j:j + side, i:i + side, ...]

		# img = imresize(img, [height, width])
		img = np.array(Image.fromarray(img).resize((height, width)))
		return img
	
	def __getitem__(self, index):
		from_path = self.source_paths[0][index]  # 全地址：root_dir + sub_dir
		from_path_rel = self.source_paths[1][index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		to_path = self.target_paths[0][index]
		to_path_rel = self.target_paths[1][index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		img_size = to_im.size()[1:]

		if self._use_mask:
			if self.mask_paths is None:
				mask = RandomMask(img_size[-1], hole_range=self.hole_range)  # hole as 0, reserved as 1
				mask = np.ones(mask.shape) - mask  # hole as 1, reserved as 0
				mask = torch.tensor(mask, dtype=torch.float)
			else:
				mask_path = self.mask_paths[0][index]
				mask = imread(mask_path)
				mask = self.resize(mask, self.opts.fineSize, self.opts.fineSize)
				if len(mask.shape) < 3:
					mask = mask[..., np.newaxis]
					mask = np.tile(mask, (1,1,3))
				else:
					mask = mask[:, :, 0]
					mask = mask[..., np.newaxis]
					mask = np.tile(mask, (1,1,3))
				mask = transforms.ToTensor()(mask)
		else:
			mask = np.ones([1, img_size[-1], img_size[-1]])
			mask = torch.tensor(mask, dtype=torch.float)

		if self.return_name:
			name = os.path.basename(to_path)
		else:
			name = None

		data = {
			'name': name,
			'mask': mask,
			'from_im': to_im,
			'condition': from_im,
		}

		return data