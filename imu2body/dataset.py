# Copyright (c) Facebook, Inc. and its affiliates.
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

import numpy as np
import pickle
import torch
from fairmotion.utils import constants
from IPython import embed
from imu2body.preprocess import load_data as load_amass_data
# from preprocess_bvh import *
from pytorch3d import transforms
import constants.motion_data as motion_constants 
from torch.utils.data import Dataset, DataLoader
from interaction.contact import *

class MotionData(Dataset):
	def __init__(self, dataset_path="", device="cuda", data=None, base_dir="", mean=None, std=None, debug=False):
		"""
		Args:
			bvh_path (string): Path to the bvh files.
			seq_len (int): The max len of the sequence for interpolation.
		"""

		self.debug = debug 

		# if isinstance(dataset_path, list):
		# 	print("got a list of pkl files")
		# 	dict_list = []
		# 	for dataset_file in dataset_path:
		# 		print(f"loading {dataset_file}")
		# 		dict_list.append(pickle.load(open(dataset_file, "rb")))
		# 	# append dict list to create data
		# 	key_list = dict_list[0].keys()
		# 	self.data = {}
		# 	for key in key_list:
		# 		dict_list_by_key = []
		# 		for i in range(len(dict_list)):
		# 			dict_list_by_key.append(dict_list[i][key])
		# 		self.data[key] = np.concatenate(dict_list_by_key, axis=0)
		# 		# print(f"finish loading key: {key}")
		# 	del dict_list
		if isinstance(dataset_path, list):
			print("IMU2Body: got a list of pkl files")

			self.data = {}

			for dataset_file in dataset_path:
				print(f"loading {dataset_file}")				
				with open(dataset_file, "rb") as file:
					current_dict = pickle.load(file)
				if not self.data:
					for key in current_dict.keys():
						self.data[key] = []
				for key in self.data.keys():
					self.data[key].append(current_dict[key])				
				del current_dict

			for key in self.data.keys():
				self.data[key] = np.concatenate(self.data[key], axis=0)

		elif 'pkl' in dataset_path:
			print(f"loading {dataset_path}")
			with open(dataset_path, "rb") as file:
				self.data = pickle.load(file)
		elif 'npz' in dataset_path:
			self.data = load_amass_data([dataset_path], base_dir="../data/amass/")
			
		# load dimension info
		self.load_data_dict()

		# set x_mean and x_std for pos scaling
		global_p = self.data['global_p']
		x_mean = np.mean(global_p.reshape([global_p.shape[0], global_p.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
		x_std = np.std(global_p.reshape([global_p.shape[0], global_p.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
		self.data['x_mean'] = x_mean
		self.data['x_std'] = x_std
		
		# normalize 
		if mean is None or std is None:
			self.mean = np.mean(self.data['input_seq'], axis=(0,1))
			self.std = np.std(self.data['input_seq'], axis=(0,1))
		else:
			self.mean = mean
			self.std = std

		self.device = device
		
	def __len__(self):
		return self.num_frames

	def __getitem__(self, idx):
		idx_ = None
		if self.debug:
			idx_ = 0
		else:
			idx_ = idx

		norm_input_seq = (self.data['input_seq'][idx_] - self.mean) / (
			self.std + constants.EPSILON
		)

		sample = {}
		sample['input_seq'] = norm_input_seq.astype(dtype=np.float32)
		sample['mid_seq'] = self.data['mid_seq'][idx_].astype(dtype=np.float32)
		sample['tgt_seq'] = self.data['tgt_seq'][idx_].astype(dtype=np.float32)

		sample['global_p'] = self.data['global_p'][idx_].astype(dtype=np.float32)
		sample['root'] = self.data['root'][idx_].astype(dtype=np.float32)

		# this is for testing and visualization		
		sample['local_rot'] = self.data['local_rot'][idx_].astype(dtype=np.float32)
		sample['head_start'] = self.data['head_start'][idx_].astype(dtype=np.float32)
		sample['contact_label'] = self.data['contact_label'][idx_].astype(dtype=np.float32)

		return sample

	def get_x_mean_and_std(self):
		return self.data['x_mean'], self.data['x_std']
	
	def get_seq_length(self):
		return motion_constants.preprocess_window

	def load_data_dict(self):
		self.num_frames, seq_len, input_seq_dim = self.data['input_seq'].shape
		assert seq_len == motion_constants.preprocess_window, "seq length should be same as window size in preprocessing! check preprocess.py"
		
		mid_seq_dim = self.data['mid_seq'].shape[2]
		output_seq_dim = self.data['tgt_seq'].shape[2]

		self.dim_dict = {}
		self.dim_dict['input_dim'] = input_seq_dim
		self.dim_dict['mid_dim'] = mid_seq_dim
		self.dim_dict['output_dim'] = output_seq_dim

	def get_data_dict(self):
		return self.dim_dict	



class CustomMotionData(Dataset):
	def __init__(self, motion_clip_path, custom_config, mean, std, device="cuda", debug=False):
		"""
		Args:
			bvh_path (string): Path to the bvh files.
			seq_len (int): The max len of the sequence for interpolation.
		"""

		base_dir = "../data/amass/"
		if 'npz' not in motion_clip_path:
			base_dir = ""

		self.data, _ = load_amass_data(base_dir=base_dir, file_list=[motion_clip_path], custom_config=custom_config)		

		# load dimension info
		self.load_data_dict()

		self.debug = debug 

		self.mean = mean
		self.std = std 
		
		self.device = device
		
		self.config = custom_config

		# contact
		self.contact = {}
		self.contact[0] = 0.0


	def __len__(self):
		return self.num_frames

	def __getitem__(self, idx):
		idx_ = None
		if self.debug:
			idx_ = 0
		else:
			idx_ = idx

		# apply contact 
		frame_idx_from_start = self.config['offset'] * idx 
		frame_key, contact_height = get_height_offset_current_frame(self.contact, frame_idx_from_start)

		self.data['input_seq'][idx_][:,0] -= contact_height
		norm_input_seq = (self.data['input_seq'][idx_] - self.mean) / (
			self.std + constants.EPSILON
		)

		sample = {}
		sample['input_seq'] = norm_input_seq.astype(dtype=np.float32)
		sample['mid_seq'] = self.data['mid_seq'][idx_].astype(dtype=np.float32)
		sample['tgt_seq'] = self.data['tgt_seq'][idx_].astype(dtype=np.float32)

		sample['global_p'] = self.data['global_p'][idx_].astype(dtype=np.float32)

		sample['local_rot'] = self.data['local_rot'][idx_].astype(dtype=np.float32)
		sample['head_start'] = self.data['head_start'][idx_].astype(dtype=np.float32)

		return sample
	
	def get_x_mean_and_std(self):
		return self.data['x_mean'], self.data['x_std']
	
	def get_seq_length(self):
		return motion_constants.preprocess_window

	def load_data_dict(self):
		self.num_frames, seq_len, input_seq_dim = self.data['input_seq'].shape
		assert seq_len == motion_constants.preprocess_window, "seq length should be same as window size in preprocessing! check preprocess.py"
		
		mid_seq_dim = self.data['mid_seq'].shape[2]
		output_seq_dim = self.data['tgt_seq'].shape[2]

		self.dim_dict = {}
		self.dim_dict['input_dim'] = input_seq_dim
		self.dim_dict['mid_dim'] = mid_seq_dim
		self.dim_dict['output_dim'] = output_seq_dim

	def get_data_dict(self):
		return self.dim_dict	


class RealMotionData(Dataset):
	def __init__(self, input_dict, mean, std, custom_config=None, device="cuda", debug=False):
		"""
		Args:
			bvh_path (string): Path to the bvh files.
			seq_len (int): The max len of the sequence for interpolation.
		"""

		# load dimension info
		self.data = input_dict
		self.load_data_dict()

		self.debug = debug 

		self.mean = mean
		self.std = std 
		
		self.device = device
		if custom_config is not None:
			self.config = custom_config 

		# contact
		self.contact = {}
		self.contact[0] = 0.0


	def __len__(self):
		return self.num_frames

	def __getitem__(self, idx):
		idx_ = None
		if self.debug:
			idx_ = 0
		else:
			idx_ = idx

		# apply contact 
		frame_idx_from_start = self.config['offset'] * idx 
		frame_key, contact_height = get_height_offset_current_frame(self.contact, frame_idx_from_start)


		self.data['input_seq'][idx_][:,0] -= contact_height

		norm_input_seq = (self.data['input_seq'][idx_] - self.mean) / (
			self.std + constants.EPSILON
		)

		# real data does not have gt!
		sample = {}
		sample['input_seq'] = norm_input_seq.astype(dtype=np.float32)
		sample['head_start'] = self.data['head_start'][idx_].astype(dtype=np.float32)

		return sample
	
	
	def get_seq_length(self):
		return motion_constants.preprocess_window

	def load_data_dict(self):
		self.num_frames, seq_len, input_seq_dim = self.data['input_seq'].shape
		assert seq_len == motion_constants.preprocess_window, "seq length should be same as window size in preprocessing!"
		

		self.dim_dict = {}
		self.dim_dict['input_dim'] = input_seq_dim
		self.dim_dict['mid_dim'] = 12
		self.dim_dict['output_dim'] = 135

	def get_data_dict(self):
		return self.dim_dict	
	

def get_loader(
	dataset_path,
	batch_size=100,
	device="cuda",
	mean=None,
	std=None,
	shuffle=False,
	drop_last=True
):
	"""Returns data loader for custom dataset.
	Args:
		dataset_path: path to pickled numpy dataset
		device: Device in which data is loaded -- 'cpu' or 'cuda'
		batch_size: mini-batch size.
	Returns:
		data_loader: data loader.
	"""
	dataset = MotionData(dataset_path=dataset_path, device=device, mean=mean, std=std)

	data_loader = DataLoader(
		dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8,  drop_last=drop_last
	)
	return data_loader

def get_custom_loader(
	motion_clip_path,
	custom_config,
	mean,
	std,
	device="cuda"
):
	"""Returns data loader for custom motion clip
	Args:
		dataset_path: path to pickled numpy dataset
		device: Device in which data is loaded -- 'cpu' or 'cuda'
		batch_size: mini-batch size.
	Returns:
		data_loader: data loader.
	"""
	dataset = CustomMotionData(motion_clip_path=motion_clip_path, device=device, custom_config=custom_config, mean=mean, std=std)

	data_loader = DataLoader(
		dataset=dataset, batch_size=1, shuffle=False, num_workers=8,  drop_last=False
	)
	return data_loader

def get_realdata_loader(
	input_dict,
	custom_config,
	mean,
	std,
	device="cuda"
):
	"""Returns data loader for custom motion clip
	Args:
		dataset_path: path to pickled numpy dataset
		device: Device in which data is loaded -- 'cpu' or 'cuda'
		batch_size: mini-batch size.
	Returns:
		data_loader: data loader.
	"""
	dataset = RealMotionData(input_dict=input_dict, custom_config=custom_config, device=device, mean=mean, std=std)

	# data loader
	data_loader = DataLoader(
		dataset=dataset, batch_size=1, shuffle=False, num_workers=8,  drop_last=False
	)
	return data_loader

if __name__=="__main__":
	# test when list of train.pkl files are given
	train_dataset = MotionData(["./data_preprocess_v2/train_1.pkl", "./data_preprocess_v2/train_2.pkl"], device="cuda") 
	fnames = ["test", "validation"]
	datasets = {}
	for fname in fnames:
		print(fname)
		datasets[fname] = MotionData(f"./data_preprocess_v2/{fname}.pkl", device="cuda")
	# fnames = ['custom']
	# datasets = {}
	# for fname in fnames:
	# 	datasets[fnames] = MotionData(f"./data/preprocess_norm/{fname}.pkl", "cpu")
	# lafan_data = MotionData('./data/lafan_mxm/', device="cpu")