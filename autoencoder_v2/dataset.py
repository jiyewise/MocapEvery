
import numpy as np
import pickle
import torch
from fairmotion.utils import constants
from IPython import embed
from autoencoder_v2.preprocess import load_data_from_amass_autoencoder as load_amass_data
from pytorch3d import transforms

from torch.utils.data import Dataset, DataLoader
import constants.motion_data as motion_constants 
from copy import deepcopy

class AEMotionData(Dataset):
	def __init__(self, dataset_path="", device="cuda", data=None, mean=None, std=None, debug=False, use_norm=False):
		"""
		Args:
			bvh_path (string): Path to the bvh files.
			seq_len (int): The max len of the sequence for interpolation.
		"""

		self.debug = debug 
		self.use_norm = use_norm


		if isinstance(dataset_path, list):
			print("AE: got a list of pkl files")

			# Initialize an empty dictionary to hold the concatenated data
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

			# Concatenate the lists in the data dictionary
			for key in self.data.keys():
				self.data[key] = np.concatenate(self.data[key], axis=0)

			
		elif 'pkl' in dataset_path:
			print(f"loading {dataset_path}")
			with open(dataset_path, "rb") as file:
				self.data = pickle.load(file)
		elif ('npz' in dataset_path) or ('bvh' in dataset_path):
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

		norm_seq = deepcopy(self.data['input_seq'][idx_])
		if self.use_norm:
			norm_seq[:,2:2+5] = (norm_seq[:,2:2+5] - self.mean[2:2+5]) / (
				self.std[2:2+5] + constants.EPSILON
			) # only normalize root pos
		else:
			norm_seq[:,2:2+3] = (norm_seq[:,2:2+3] - self.mean[2:2+3]) / (
				self.std[2:2+3] + constants.EPSILON
			) # only normalize root pos

		sample = {}
		sample['input_seq'] = norm_seq
		# sample['input_seq'] = self.data['input_seq'][idx_]
		sample['tgt_seq'] = self.data['input_seq'][idx_]

		sample['global_p'] = self.data['global_p'][idx_]
		sample['root'] = self.data['root'][idx_]
		
		# this is for testing and visualization		
		sample['local_rot'] = self.data['local_rot'][idx_].astype(dtype=np.float32)
		sample['root_start'] = self.data['root_start'][idx_].astype(dtype=np.float32)
		sample['contact_label'] = self.data['contact_label'][idx_].astype(dtype=np.float32)

		# sample['local_q'] = self.data['local_q'][idx_]
		# sample['contact'] = self.data['contact'][idx_]

		return sample

	def get_x_mean_and_std(self):
		return self.data['x_mean'], self.data['x_std']

	def get_seq_dim(self):
		return self.seq_dim
	
	def get_seq_length(self):
		return self.seq_len

	def load_data_dict(self):
		self.num_frames, seq_len, input_seq_dim = self.data['input_seq'].shape
		# assert seq_len == (motion_constants.preprocess_window * 3), "seq length should be same as window size in preprocessing! check preprocess.py"
		
		# mid_seq_dim = self.data['mid_seq'].shape[2]
		# output_seq_dim = self.data['tgt_seq'].shape[2]

		self.dim_dict = {}
		self.dim_dict['input_dim'] = input_seq_dim
		# self.dim_dict['mid_dim'] = mid_seq_dim
		self.dim_dict['output_dim'] = input_seq_dim

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
		data_loader: data loader for custom dataset.
	"""
	# build a custom dataset
	dataset = AEMotionData(dataset_path=dataset_path, device=device, mean=mean, std=std)

	# data loader for custom dataset
	# this will return (src_seqs, tgt_seqs) for each iteration
	data_loader = DataLoader(
		dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8,  drop_last=drop_last
	)
	return data_loader

# def get_loader_from_data(
# 	data,
# 	base_dir="",
# 	seq_len=80,
# 	batch_size=100,
# 	device="cuda",
# 	mean=None,
# 	std=None,
# 	shuffle=False,
# 	drop_last=True
# ):
# 	"""Returns data loader for custom dataset.
# 	Args:
# 		dataset_path: path to pickled numpy dataset
# 		device: Device in which data is loaded -- 'cpu' or 'cuda'
# 		batch_size: mini-batch size.
# 	Returns:
# 		data_loader: data loader for custom dataset.
# 	"""
# 	# build a custom dataset
# 	dataset = MotionData(data=data, base_dir=base_dir, device=device, mean=mean, std=std, seq_len=seq_len)

# 	# data loader for custom dataset
# 	# this will return (src_seqs, tgt_seqs) for each iteration
# 	data_loader = DataLoader(
# 		dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8,  drop_last=drop_last
# 	)
# 	return data_loader


if __name__=="__main__":
	fnames = ['custom']
	datasets = {}
	for fname in fnames:
		datasets[fnames] = AEMotionData(f"./data/preprocess_norm/{fname}.pkl", "cpu")
	# lafan_data = MotionData('./data/lafan_mxm/', device="cpu")