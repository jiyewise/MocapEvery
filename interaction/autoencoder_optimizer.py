import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
import argparse
from copy import deepcopy
import logging
from select import select
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from IPython import embed 
from autoencoder_v2.model import Conv3AutoEncoder, ConvAutoencoder, Seq2Seq, LSTMDecoder, LSTMEncoder
import yaml
from autoencoder_v2.run import *
from autoencoder_v2.dataset import *
from pytorch3d import transforms
import dadaptation
from tensorboardX import SummaryWriter
from imu2body.imu import _syn_acc_torch
logging.basicConfig(
	format="[%(asctime)s] %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	level=logging.INFO,
)

class LatentOptimizeConv(nn.Module):
	def __init__(self):
		super(LatentOptimizeConv, self).__init__()

	def set_params(self, latent):
		self.latent = nn.Parameter(latent)

	def forward(self, model):
		# the model is freezed
		output = model.decode(self.latent)
		return output

class AutoEncoderOptim(object):
	def __init__(self, ae_network) -> None:
		self.autoencoder = ae_network 
		self.config = self.autoencoder.config
		self.device = self.autoencoder.device 
		self.use_norm = self.autoencoder.norm 

		# x_mean and x_std
		self.x_mean = self.autoencoder.x_mean
		self.x_std = self.autoencoder.x_std 

		# adjust loss scale
		self.loss_weight = {}
		self.loss_weight["imu"] = 0.03
		self.loss_weight["reproj"] = 0.8
		self.loss_weight["reg"] = 2
		self.loss_weight["hand_reg"] = 0.3
		self.loss_weight["cvel"] = 5
		self.loss_weight["hand_vel"] = 10

		self.load_latent_module()
	
	def load_latent_module(self):
		if self.config['model']['type'] == "conv3":
			self.latent_module = LatentOptimizeConv().to(self.device)
		else:
			raise NotImplementedError("conv3 supported")
	
	def load_optimizer(self):
		self.optimizer = optim.Adam(self.latent_module.parameters(), lr=0.0007, betas=[0.8, 0.82])	
		for param in self.autoencoder.model.parameters():
			param.requires_grad = False
		
	def set_loss_std(self, loss_std_dict):
		self.loss_std = deepcopy(loss_std_dict)
		for loss_key in self.loss_std:
			self.loss_std[loss_key] = torch.from_numpy(self.loss_std[loss_key]).to(self.device).float()

	def normalize_input(self, input_seq):
		norm_seq = deepcopy(input_seq[0])
		mean, std = self.autoencoder.mean, self.autoencoder.std
		if self.use_norm:
			norm_seq[:,2:2+5] = (norm_seq[:,2:2+5] - mean[2:2+5]) / (
				std[2:2+5] + constants.EPSILON
			) # only normalize root pos
		else:
			norm_seq[:,2:2+3] = (norm_seq[:,2:2+3] - mean[2:2+3]) / (
				std[2:2+3] + constants.EPSILON
			) # only normalize root pos
		return norm_seq[np.newaxis, ...]
	
	def get_reprojection_loss(self, re_dict, output_pos):
		target_2d = torch.from_numpy(re_dict['target_2d']).to(self.device).float() # torch.Size([2])
		cam_ext = torch.from_numpy(re_dict['normalized_cam_ext']).to(self.device).float() # torch.Size([4, 4])
		joint_idx = re_dict['joint_idx']
		frame = re_dict['frame']
		lr = motion_constants.imu_hand_joint_idx.index(joint_idx)
		reproj_loss_std = self.loss_std['reproject'][lr]

		joint_pos_3d = output_pos[0,frame,joint_idx]

		# reproject 
		cam_ext_inv = invert_T_torch(cam_ext)
		point_in_cam_local = (cam_ext_inv[:3,:3] @ joint_pos_3d.unsqueeze(-1)).squeeze(-1) + cam_ext_inv[:3,3]
		reprojected_point = (self.cam_intrinsics @ point_in_cam_local.unsqueeze(-1)).squeeze(-1)
		reprojected_point = reprojected_point / reprojected_point[...,2].unsqueeze(-1)
		
		reproject_loss = torch.mean(torch.abs(target_2d - reprojected_point[:2])) / reproj_loss_std

		return reproject_loss


	def latent_optimize(self, input_data_dict, epoch_num=500):
		logging.info(f"optimization start for frame {input_data_dict['start_frame']}")
		input_seq = input_data_dict['input_seq']
		# normalize 
		mean, std = self.autoencoder.mean, self.autoencoder.std 
		norm_seq = self.normalize_input(input_seq=input_seq)

		# convert to torch 
		norm_seq_torch = torch.from_numpy(norm_seq).to(self.device)

		self.autoencoder.model.eval()
		latent = self.autoencoder.model.encode(norm_seq_torch) # get latent vector
		self.latent_module.set_params(latent=latent)
		self.initial_latent = latent
		self.load_optimizer()

		self.loss_total = None
		# constants
		# self.initial_loss = 100
		self.prev_loss = 100
		self.no_improvement = 0
		self.cam_intrinsics = torch.from_numpy(input_data_dict['cam_int']).to(self.device).float()

		for epoch in range(epoch_num):
			output_seq = self.latent_module(self.autoencoder.model)
			batch, seq_len, _ = output_seq.shape
	
			if seq_len != 80: # autoencoder window size
				return None, None, None
			
			# reconstruct 
			skel = self.autoencoder.skel_offset.repeat(batch, seq_len, 1, 1)
			output_root = output_seq[...,2:2+3]
			output_joint_rot = output_seq[...,2+3:]
			output_joint_rot = output_joint_rot.reshape(batch, seq_len, -1, 6)
			output_joint_rotmat = transforms.rotation_6d_to_matrix(output_joint_rot) # [1, 80, 22, 3, 3]

			output_pos = rot_matrix_fk_tensor(output_joint_rotmat, output_root, skel, self.autoencoder.skel_parent) # [1, seq_len, 22, 3]

			# target loss 
			reproj_target_loss = torch.tensor(0.0).to(self.device)
			if len(input_data_dict['reproj_target']) > 0:
				for re_dict in input_data_dict['reproj_target']:
					loss_per_target = self.get_reprojection_loss(re_dict=re_dict, output_pos=output_pos)
					reproj_target_loss += loss_per_target

				reproj_target_loss /= float(len(input_data_dict['reproj_target']))
				if torch.isnan(reproj_target_loss):
					print("reprojection error is nan!")
					embed()

			# reg loss 
			frame_start, frame_end = input_data_dict['frame_start_end']
			gt_global_pos = torch.from_numpy(input_data_dict['global_p']).to(self.device).float()
			pos_diff = torch.abs(gt_global_pos[:, ...] - output_pos) / self.autoencoder.x_std # [1, 80, 22, 3]
				
			# reg_loss = torch.mean(torch.abs(pos_diff))
			reg_loss = torch.mean(torch.abs(pos_diff[...,:18,:]))
			hand_reg_loss = torch.mean(torch.abs(pos_diff[...,18:22,:]))

			# contact velocity loss
			toe_vel = output_pos[0,1:,motion_constants.toe_joints_idx,:] - output_pos[0,:-1,motion_constants.toe_joints_idx,:]
			toe_vel = torch.linalg.norm(torch.cat((toe_vel[0:1, :, :], toe_vel), dim=0),dim=-1)
			c_lr = torch.from_numpy(input_data_dict['contact_label'][frame_start:frame_end]).to(self.device).float()
			# contact_vel = (c_lr * toe_vel) / self.loss_std['contact']
			contact_vel = (c_lr * toe_vel)
			contact_vel_loss = torch.mean(contact_vel)

			self.loss_total = self.loss_weight["cvel"] * contact_vel_loss + \
								self.loss_weight["reproj"] * reproj_target_loss + \
								self.loss_weight["reg"] * reg_loss + \
								self.loss_weight["hand_reg"] * hand_reg_loss 
								# self.loss_weight["hand_vel"] * vel_loss

			if epoch == 0:
				self.initial_loss = self.loss_total.item()

			print(f"current step: {epoch} no_imp: {self.no_improvement} total loss: {self.loss_total.item()} init: {self.initial_loss} reproj: {reproj_target_loss} reg:{reg_loss} cvel: {contact_vel_loss}", end='\r')

			if self.loss_total.item() < 1:
				if self.loss_total.item() >= self.prev_loss:
					self.no_improvement += 1
				else:				
					self.prev_loss = self.loss_total.item()			
					if self.no_improvement > 20:
						break
			
			self.loss_total.backward()
			torch.nn.utils.clip_grad_norm_(self.latent_module.parameters(), 0.5)
			self.optimizer.step()
		
		print("")


		loss_record = {}
		loss_record['total'] = self.loss_total
		# loss_record['imu'] = imu_loss
		loss_record['reproj'] = reproj_target_loss
		loss_record['reg'] = reg_loss
		loss_record['hand_reg'] = hand_reg_loss
		loss_record['cvel'] = contact_vel_loss
		loss_record['init'] = self.initial_loss

		# embed()
		output_joint_rot = transforms.rotation_6d_to_matrix(output_joint_rot)
		# get global T
		gr_rotmat, gp_pos = rot_matrix_fk_tensor(output_joint_rotmat, output_root, skel, self.autoencoder.skel_parent, return_T=True)
		batch, seq_len, joint_num, _ = gp_pos.shape 
		global_T = np.zeros(shape=(seq_len, joint_num, 4, 4))
		global_T[...,:3,:3] = gr_rotmat.detach().cpu().numpy()
		global_T[...,:3,3] = gp_pos.detach().cpu().numpy()


		rd = AERenderData( output_root=output_root[0].detach().cpu().numpy(),\
		output_rot=output_joint_rot[0].detach().cpu().numpy(), \
		use_gt=False)
		logging.info(f"optimization done for frame {input_data_dict['start_frame']}")

		# TODO replace to root start - check working
		start_T = input_data_dict['root_start']
		rd.convert_to_matrix(start_T=start_T[0])
		
		global_T = start_T @ global_T
		return rd.output_T, global_T[0], loss_record


	# def postprocess_optimized_result(self):
	# 	return 