import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
from IPython import embed

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import dateutil.parser
from scipy import ndimage 
from copy import deepcopy
from fairmotion.ops import conversions
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from IPython import embed
from fairmotion.ops import conversions
import torch
import torch.nn as nn
import torch.optim as optim
from imu2body.functions import *

import pytorch3d
from tqdm import tqdm
import fairmotion.utils as utils
import glob 
import constants.motion_data as motion_constants
from fairmotion.ops import math as fmath
from realdata.utils import *
import constants.path as path_constants

from realdata.utils import * 
from imu2body.imu import _syn_acc

class CameraHeadSignal2(object):
	def __init__(self, dir) -> None:
		self.dir = dir 
		self.empty = True
		self.load_sfm_result()


	def load_sfm_result(self):
		self.load_intrinsics()
		self.load_cam_traj()
		if self.empty:
			return
		self.set_timecode()

	def load_intrinsics(self):
		self.intrinsics = np.eye(3)
		undistort_cam_int_path = os.path.join(path_constants.DATA_PATH, f"{self.dir}/sfm/undistort_calib.txt")
		calib = np.loadtxt(undistort_cam_int_path, delimiter=" ")
		fx, fy, cx, cy = calib[:4]
		self.intrinsics[0,0] = fx
		self.intrinsics[0,2] = cx
		self.intrinsics[1,1] = fy
		self.intrinsics[1,2] = cy


	def load_cam_traj(self, custom_path=None):
		cam_traj_path = os.path.join(path_constants.DATA_PATH, f"{self.dir}/cam_traj.pkl")
		if custom_path:
			cam_traj_path = custom_path
		if not os.path.exists(cam_traj_path):
			print("Camera trajectory file does not exist!")
			return 
		if sys.version_info < (3, 8):
			import pickle5 as pickle 
		else:
			import pickle
		cam_traj = pickle.load(open(cam_traj_path, "rb"))

		cam_ext_p = cam_traj[:,:3]
		cam_ext_q = cam_traj[:,3:]
		cam_ext_T = conversions.Qp2T(cam_ext_q, cam_ext_p)
		
		self.cam_traj = cam_ext_T
		print("reading camera trajectory done")
		self.empty = False

		# remove outliers
		self.remove_outliers()


	def load_undistorted_img_filenames(self):
		img_dir = os.path.join(path_constants.DATA_PATH, f"{self.dir}/sfm/undistorted_images")

		files = glob.glob(os.path.join(img_dir, '*'))
		image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
		image_files.sort()

		self.image_filename_list = image_files
		self.mocap_img_list = self.image_filename_list[self.mocap_frames[0]:self.mocap_frames[-1]+1]

	def set_time_config(self, calib_dict, seq_len=None):
		self.set_timecode(calib_dict['shutter'], seq_len=seq_len)
		# get by frames
		self.align_config = {}
		for key in calib_dict:
			if key not in ['x', 'y', 'clap']: continue
			timecode_range = calib_dict[key]
			self.align_config[key] = []
			for tc in timecode_range:
				self.align_config[key].append((np.abs(self.timecode - tc)).argmin())


	def set_timecode(self, shutter=0.0, seq_len=None):
		if seq_len is None:
			seq_len, _, _ = self.cam_traj.shape 
		self.timecode = [shutter * 1000]
		current = 0.0
		for _ in range(1, seq_len):
			current = 34 if _ % 2 else 33
			self.timecode.append(current + self.timecode[-1])
		
		self.timecode = np.array(self.timecode)
		self.timecode *= 0.001 # into ms
	
	def calib_tpose(self):
		tpose_head_T = self.cam_traj[self.align_config['clap'][1]] @ motion_constants.cam2head
		tpose_head_R = tpose_head_T[:3,:3]
		head2tpose = tpose_head_R.transpose() @ motion_constants.head_ori_tpose

		self.cam2head_T = motion_constants.cam2head @ conversions.R2T(head2tpose)
	

	# remove outliers and fill in by interpolation
	def remove_outliers(self):
		self.removed_cam_traj = []
		self.removed_cam_traj.append(self.cam_traj[0])

		vel_list = []
		nan_idx_per_neighbor = {}
		nan_close_per_start_neighbor = {}
		not_nan_neighbor = 0
		
		acc = _syn_acc(self.cam_traj[:,:3,3][:,np.newaxis,:])
		acc = np.linalg.norm(acc, axis=-1)

		prev_nan = False
		for idx in range(1, len(self.cam_traj)):
			acc_val = acc[idx]
			if acc_val < 3:
				self.removed_cam_traj.append(self.cam_traj[idx])
				if prev_nan:
					nan_close_per_start_neighbor[not_nan_neighbor] = idx
					prev_nan = False
				not_nan_neighbor = idx
			else:
				self.removed_cam_traj.append(np.nan)
				if not_nan_neighbor not in nan_idx_per_neighbor:
					nan_idx_per_neighbor[not_nan_neighbor] = []
				nan_idx_per_neighbor[not_nan_neighbor].append(idx)
				prev_nan = True
			
		# fillin outliers 
		for start_neighbor in nan_idx_per_neighbor:
			nan_idx_list = nan_idx_per_neighbor[start_neighbor]
			end_neighbor = nan_idx_list[-1] + 1
			if end_neighbor >= self.cam_traj.shape[0]:
				continue
			for nan_idx in nan_idx_list:
				self.cam_traj[nan_idx] = self.cam_traj[start_neighbor]
				weight = float(nan_idx - start_neighbor) / float(end_neighbor - start_neighbor)
				inter_R = fmath.slerp(self.cam_traj[start_neighbor,:3,:3], self.cam_traj[end_neighbor,:3,:3], weight)
				inter_p = fmath.lerp(self.cam_traj[start_neighbor,:3,3], self.cam_traj[end_neighbor,:3,3], weight)
				self.cam_traj[nan_idx] = conversions.Rp2T(inter_R, inter_p)


	# set motion region
	def set_data_for_network(self, start_timecode, end_timecode=None):
		self.calib_tpose()
		mocap_frames = time_range_into_frames(self.timecode, start_timecode=start_timecode, end_timecode=end_timecode)
		self.mocap_timecode = self.timecode[mocap_frames]
		self.mocap_cam = self.cam_traj[mocap_frames]
		self.mocap_frames = mocap_frames
		self.load_undistorted_img_filenames()
		
		self.mocap_cam2head = np.einsum('ijk,km->ijm', self.mocap_cam, self.cam2head_T)

		height = self.mocap_cam2head[0,2,3]
		# match amass
		ratio = 1.494/height
		self.mocap_cam2head[:,2,3] *= ratio

	def get_time_range(self):
		return self.mocap_timecode[0], self.mocap_timecode[-1], len(self.mocap_cam)

	def load_pcd_result(self):
		pcd_path = os.path.join(path_constants.DATA_PATH, f"{self.dir}/sfm/pointcloud/pcd_results.ply")
		self.pcd = o3d.io.read_point_cloud(pcd_path)
		self.align_pcd()

	def align_pcd(self):
		s = self.scale

		scale_matrix = [[s, 0, 0, 0],
						[0, s, 0, 0],
						[0, 0, s, 0],
						[0, 0, 0, 1]]
		scale_matrix = np.array(scale_matrix)
		T = np.dot(self.origin_mapper, scale_matrix)
		self.pcd.transform(T)

		# height 
		height_T = np.eye(4)
		height_T[2,3] = self.height_diff  # make floor stick to foot.. 

		self.pcd.transform(height_T)

	# use zx for matching scale and origin
	def match_scale_origin_2d(self, config):
		y_start, y_end = self.cam_traj[self.align_config['y'][0]], self.cam_traj[self.align_config['y'][1]]
		x_start, x_end = self.cam_traj[self.align_config['x'][0]], self.cam_traj[self.align_config['x'][1]]

		y_vec = y_end[:3,3] - y_start[:3,3]
		x_vec = x_end[:3,3] - x_start[:3,3]

		y_dist = np.linalg.norm(y_vec)
		x_dist = np.linalg.norm(x_vec)

		x_dist_gt = config['dist'][0]
		y_dist_gt = config['dist'][1]

		scale = (y_dist_gt/y_dist) * 0.5 + (x_dist_gt/x_dist)* 0.5  # TODO connect this to config or so ... 

		print(f"scale mean: {scale}")

		# embed()
		# match scale of cam
		self.cam_traj[...,:3,3] = scale * self.cam_traj[...,:3,3]

		# match origin
		self.origin_optim = CameraOriginOptim2(xvec=scale*x_vec, yvec=scale*y_vec)
		self.optimizer = optim.Adam(self.origin_optim.parameters(), lr=0.007, betas=[0.8, 0.82])	

		is_mid_checkpoint = False
		mid_R = None
		mid_loss = None
		
		# to check improvement
		prev_loss = 100
		is_improvement = False
		for i in range(1000):
			loss = self.origin_optim.forward()
			# print(f"loss: {loss.item()} step: {i}", end='\r')
			print(f"loss: {loss.item()} step: {i}")
			if loss.item() < 0.2:
				if loss.item() < prev_loss:
					is_mid_checkpoint = True 
					mid_R = self.origin_optim.get_current_R()
					mid_loss = loss.item()
					prev_loss = mid_loss
			if loss.item() < 0.1:
				is_mid_checkpoint = True
				mid_R = self.origin_optim.get_current_R()
				mid_loss = loss.item()
				break # early stopping
			if i > 300 and loss.item() < 0.35:
				mid_loss = loss.item()
				break
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		if is_mid_checkpoint:
			origin_mapper = mid_R
		else:
			origin_mapper = self.origin_optim.get_current_R()
		origin_mapper = conversions.R2T(origin_mapper.detach().cpu().numpy())

		self.cam_traj = origin_mapper[np.newaxis, ...] @ self.cam_traj

		# set height 
		y_start_pos = self.cam_traj[self.align_config['y'][0]]
		height_diff = config['height'] - y_start_pos[2,3] # TODO move to config['height']

		self.cam_traj[...,2,3] += height_diff

		print("\n")

		self.scale = scale 
		self.origin_mapper = origin_mapper

		self.height_diff = height_diff


	# (used only for getting cam2head via optimization)
	def optimize_cam2head(self, config, head_tpose_seq, head_rotate_seq):
		# NOTE here the number is mocap frame standard (count after set_frame_sync is called)

		# tpose (to get rotation)
		cam_tpose_range = config['tpose']
		cam_tpose_range_R = conversions.T2R(self.cam_traj[cam_tpose_range[0]:cam_tpose_range[1]])
		head_tpose_range_R = conversions.T2R(head_tpose_seq)

		cam2head_R = cam_tpose_range_R.swapaxes(-2,-1) @ head_tpose_range_R	# TODO check
		cam2head_aa = conversions.R2A(cam2head_R)

		cam2head_aa_mean = np.mean(cam2head_aa, axis=0)
		cam2head_R_mean = conversions.A2R(cam2head_aa_mean)

		# rotate (to get translation)
		cam_range = config['rotate']
		cam_rotate_seq = self.cam_traj[cam_range[0]:cam_range[1]]
		self.cam2head_optim = Cam2HeadOptim(cam2head_R=cam2head_R_mean, head_rotate_seq=head_rotate_seq, cam_rotate_seq=cam_rotate_seq)
		self.cam2head_optimizer = optim.Adam(self.cam2head_optim.parameters(), lr=0.001, betas=[0.8, 0.82])	

		for i in range(500):

			trans_loss = self.cam2head_optim.forward()
			print(f"trans_loss: {trans_loss.item()} step: {i}")

			self.cam2head_optimizer.zero_grad()
			trans_loss.backward()
			self.cam2head_optimizer.step()
		
		cam2head_T = self.cam2head_optim.get_current_cam2head()
		cam2head_T = cam2head_T.detach().cpu().numpy()

		self.cam2head_T = cam2head_T

		print(f"cam2head: {cam2head_T}")
		self.cam_traj2head = np.einsum('ijk,km->ijm', self.cam_traj, self.cam2head_T)
		

	# the following functions are for matching gt
	# sync by timecode
	def set_timecode_offset(self, timecode_offset):
		self.timecode += timecode_offset
		self.timecode_offset_with_mocap = timecode_offset


	# sync by clap frame
	def set_frame_sync(self, frame_offset, num_frames):
		self.cam_traj = self.cam_traj[frame_offset:]
		seq_len, _, _ = self.cam_traj.shape
		if seq_len > num_frames:
			self.cam_traj = self.cam_traj[:num_frames]

		# assert len(gt_timecode) > seq_len, "Currently, camera is turned off earlier. change this when settings change"
		# self.timecode = deepcopy(gt_timecode)[:seq_len]

			

class CameraOriginOptim2(nn.Module):
	def __init__(self, xvec, yvec):
		super(CameraOriginOptim2, self).__init__()

		self.x = torch.tensor(np.array([1, 0.0, 0.0])).float()
		self.y = torch.tensor(np.array([0.0, 1, 0.0])).float()

		rot_identity = torch.tensor([1,0,0,0,1,0]).float()
		self.rotate_6d = nn.Parameter(rot_identity)

		self.set_vector(xvec, yvec)


	def set_vector(self, xvec, yvec):
		self.xvec = torch.tensor(deepcopy(xvec)).float()
		self.yvec = torch.tensor(deepcopy(yvec)).float()

	
	def forward(self):
		R = self.get_current_R()
		mapped_xvec = R @ self.xvec
		mapped_yvec = R @ self.yvec 

		height_loss = torch.abs(mapped_xvec[2]) + torch.abs(mapped_yvec[2])

		diff_Rx = fmath.R_from_vectors_torch(mapped_xvec, self.x)
		diff_Ry = fmath.R_from_vectors_torch(mapped_yvec, self.y)
		loss = (self.rotation_distance_from_identity(diff_Rx) * 5 + self.rotation_distance_from_identity(diff_Ry)) * 5 + height_loss 

		# loss = torch.mean(torch.abs(mapped_xvec - self.x)) + torch.mean(torch.abs(mapped_yvec - self.y))

		return loss
	
	def rotation_distance_from_identity(self, R):
		"""Compute the angle (in radians) between a rotation matrix and the identity matrix."""
		div = 2.0 + 1e-06 # to avoid nan
		theta = torch.acos((torch.trace(R) - 1) / div)
		return theta
		
	def get_current_R(self):
		return pytorch3d.transforms.rotation_6d_to_matrix(self.rotate_6d)


# would be used only to get cam2head transformation matrix (this is not done every time, fix as const)
class Cam2HeadOptim(nn.Module):
	def __init__(self, cam2head_R, head_rotate_seq, cam_rotate_seq) -> None:
		super(Cam2HeadOptim, self).__init__()

		# set const and config
		self.cam_seq = torch.tensor(deepcopy(cam_rotate_seq)).float() 
		self.head_seq = torch.tensor(deepcopy(head_rotate_seq)).float()

		# set parameters 
		# rot_identity = torch.tensor([1,0,0,0,1,0]).float()

		self.rotate = torch.tensor(deepcopy(cam2head_R)).float()
		self.translation = nn.Parameter(torch.rand(3))

		self.criterion = nn.L1Loss()

	def forward(self):
		
		# map cam to head
		cur_cam2head = self.get_current_cam2head()
		mapped_head = self.cam_seq @ (cur_cam2head.unsqueeze(0)) 

		# make mapped head & head seq into first frame local 
		mapped_head_start_inv = invert_T_torch(mapped_head[0:1])
		localized_mapped_head = mapped_head_start_inv @ mapped_head

		head_start_inv = invert_T_torch(self.head_seq[0:1])
		localized_head = head_start_inv @ self.head_seq

		# compare losses 
		# localized_mapped_head_rot = transforms.matrix_to_rotation_6d(localized_mapped_head[:,:3,:3])
		# localized_head_rot = transforms.matrix_to_rotation_6d(localized_head[:,:3,:3])

		# rot_loss = self.criterion(localized_mapped_head_rot, localized_head_rot)
		trans_loss = self.criterion(localized_mapped_head[:,:3,3], localized_head[:,:3,3])

		# consider height 
		height_loss = self.criterion(mapped_head[:,2,3], self.head_seq[:,2,3])
		return trans_loss * 10 + height_loss * 5
	
		# # embed()
		# loss = rot_loss + trans_loss

		# return loss, rot_loss, trans_loss
	
	def get_current_cam2head(self):
		cam2head = torch.eye(4)
		cam2head[:3, :3] =  self.rotate
		cam2head[:3, 3] = self.translation

		return cam2head
	

if __name__ == "__main__":
	camera_signal = CameraHeadSignal2("0723")