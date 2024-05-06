import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
import argparse
from copy import deepcopy
import logging
import numpy as np
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
from IPython import embed 
# from bvh import *
import yaml
from autoencoder_v2.dataset import *
from imu2body.functions import *
from pytorch3d import transforms
import dadaptation
from fairmotion.data import bvh
import imu2body.amass as amass
from tqdm import tqdm
from fairmotion.ops import conversions
import constants.motion_data as motion_constants
from autoencoder_v2.model import * 
import constants.path as path_constants
from tensorboardX import SummaryWriter
 
logging.basicConfig(
	format="[%(asctime)s] %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	level=logging.INFO,
)

change_mode_epoch = 40

class AERenderData(object):
	def __init__(self,output_root, output_rot, use_gt=True, gt_root=None, gt_rot=None):
		self.output_root = output_root # [seq_len, 3]
		self.output_rot = output_rot # [seq_len, J, 4]
		if use_gt:
			self.gt_root = gt_root # [seq_len, 3]
			self.gt_rot = gt_rot # [seq_len, J, 4]
		self.start_T = None
		self.use_gt = use_gt
		self.start_frame = None
		self.parse_frame = None

	def convert_to_T(self, rot, root):
		if rot.shape[-1] == 3:
			return rotmat_to_T_motion(rot, root)
		else:
			assert rot.shape[-1] == 4, "rotations should be either rotation matrices or quaternions"
			return quat_to_T_motion(rot, root)
	

	def convert_to_matrix(self, scale=1, start_T=None):
		self.output_root *= scale
		self.output_T = self.convert_to_T(self.output_rot, self.output_root)
		if start_T is not None:
			self.start_T = start_T
			self.output_T[...,0:1,:,:] = start_T @ self.output_T[...,0:1,:,:]	

			if self.use_gt:
				self.gt_root *= scale
				self.gt_T = self.convert_to_T(self.gt_rot, self.gt_root)
				# self.gt_T[...,0:1,:,:] = start_T @ self.gt_T[...,0:1,:,:]

	
	def set_frame(self, start_frame=None, parse_frame=None):
		self.start_frame = start_frame
		self.parse_frame = parse_frame


def ae_write_result_pkl(render_data_list, save_dir, filename="", custom_config=None, contact_manager=None):
    data = {}
    data['output'] = []
    data['gt'] = []

    for rd in render_data_list:
        start_frame = 0 if rd.start_frame is None else rd.start_frame
        parse_frame = rd.output_T.shape[0] if rd.parse_frame is None else rd.parse_frame
        data['output'].append(rd.output_T[start_frame:parse_frame, ...])
        if rd.use_gt:
            data['gt'].append(rd.gt_T)
        else:
            data['gt'].append(None)

    # dump
    if filename == "":
        now = datetime.datetime.now()
        filename = now.strftime('%m-%d-%H-%M')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    pkl_filename = os.path.join(save_dir, f"{filename}.pkl")
    with open(pkl_filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
	
def set_seeds():
	torch.manual_seed(1234)
	np.random.seed(1234)
	random.seed(1234)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


class AutoEncoder(object):
	def __init__(self, args):
		set_seeds()

		# init 
		directory = "./output/" + args.test_name + "/"
		if args.mode == "custom" and not os.path.exists(directory): # called from other module
			directory = f"{path_constants.BASE_PATH}/autoencoder_v2/output/{args.test_name}/"

		if not os.path.exists(directory):
			os.mkdir(directory)


		self.directory = directory
		self.mode = args.mode
		
		# open config file
		config_dir = "./config/"+args.config + ".yaml" if self.mode == "train" else self.directory + "config.yaml"
		self.config = yaml.safe_load(open(config_dir, 'r').read())
		self.norm = self.config['data']['norm']

		if self.mode == "train":
			os.system('cp {} {}'.format('./config/'+args.config+'.yaml', directory+'config.yaml'))

		self.data_path = self.config['data']['preprocess'] # if self.mode != "custom" else args.custom_datapath

		logging.info(f"Starting in {self.mode} mode...")

		self.set_info()
		
		# if self.mode != "custom":
		self.load_data()  
		
		self.build_network()
		if self.mode != "custom":
			self.build_optimizer()


	def set_info(self, pretrain=False):
		is_train = True if self.mode == "train" else False
		self.pretrain = pretrain if is_train else True
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		logging.info(f"Using device: {self.device}")
		
		# set information from config and args
		self.save_frequency = self.config['train']['save_frequency']

		# if self.mode in ["train", "test"]:
		self.set_skel_info() # load skeleton info (this is needed for train and test)
		
		self.log_dir = os.path.join(self.directory, "log/")
		self.model_dir = os.path.join(self.directory, "model/")

		if not os.path.exists(self.log_dir):
			os.mkdir(self.log_dir)

		if not os.path.exists(self.model_dir):
			os.mkdir(self.model_dir)

		self.train_epoch = 0

	def set_skel_info(self):

		body_model = amass.load_body_model(motion_constants.BM_PATH)
		fairmotion_skel, _ = amass.create_skeleton_from_amass_bodymodel(bm=body_model)

		self.skel = fairmotion_skel
		self.skel_offset = fairmotion_skel.get_joint_offset_list()
		self.skel_parent = fairmotion_skel.get_parent_index_list()
		
		self.ee_idx = fairmotion_skel.get_index_joint(motion_constants.EE_JOINTS)
		# self.foot_idx = fairmotion_skel.get_index_joint(motion_constants.FOOT_JOINTS) + fairmotion_skel.get_index_joint(motion_constants.toe_joints)
		# self.hand_idx = fairmotion_skel.get_index_joint(motion_constants.HAND_JOINTS)
		# # this is to solve overfitting on foot
		self.leg_idx = fairmotion_skel.get_index_joint(motion_constants.LEG_JOINTS)

		self.skel_offset = torch.from_numpy(self.skel_offset[np.newaxis, np.newaxis, ...]).to(self.device).float() 		# expand skel offset into tensor

		if self.mode == "train":
			autoencoder_window_size = motion_constants.preprocess_window * 2
			self.skel_offset = self.skel_offset.repeat(self.config['train']['batch_size'], autoencoder_window_size, 1, 1)
	
	def load_data(self):
		is_train = False
		fnames = [self.mode]
		if self.mode == "train":
			fnames.append("validation")
			is_train = True

		self.dataloader = {}
		if is_train is False:
			data = np.load(self.directory + "mean_and_std.npz")
			self.mean = data['mean']
			self.std = data['std']

			x_data = np.load(self.directory + "x_mean_and_std.npz")
			self.x_mean = x_data['mean']
			self.x_std = x_data['std']

		if self.mode == "custom":
			self.x_mean = torch.from_numpy(self.x_mean).to(self.device)
			self.x_std = torch.from_numpy(self.x_std).to(self.device).view(1, 1, motion_constants.NUM_JOINTS, 3)
			return 
		
		train_split_num = 12
		test_split_num = 2
		split_num = {}
		split_num['train'] = train_split_num
		split_num['test'] = test_split_num
		split_num['validation'] = 0

		for fname in fnames:
			if fname == "train":
				batch_size = self.config[fname]['batch_size']
				train_fnames = [os.path.join(self.data_path, f"train_{i+1}.pkl") for i in range(split_num[fname])] # TODO fix so automatically would be all read
				self.dataloader[fname] = get_loader(dataset_path=train_fnames, \
												batch_size=batch_size, \
												device=self.device, \
												shuffle=is_train)
				self.mean = self.dataloader['train'].dataset.mean
				self.std = self.dataloader['train'].dataset.std 
				np.savez(self.directory+"mean_and_std", mean=self.mean, std=self.std)

				self.x_mean, self.x_std = self.dataloader['train'].dataset.get_x_mean_and_std() 
				np.savez(self.directory+"x_mean_and_std", mean=self.x_mean, std=self.x_std)

			else: 
				split_num_cur = split_num[fname]
				if split_num_cur > 0:
					dataset_path = [os.path.join(self.data_path, f"{fname}_{i+1}.pkl") for i in range(split_num[fname])] 
				else:
					dataset_path = os.path.join(self.data_path, f"{fname}.pkl")
				# dataset_path = os.path.join(self.data_path, f"validation.pkl")
				batch_size = self.config['train']['batch_size'] if fname == "validation" else self.config['test']['batch_size']
				# batch_size = self.config['train']['batch_size']
				self.dataloader[fname] = get_loader(dataset_path=dataset_path, \
												batch_size=batch_size, \
												device=self.device, \
												mean = self.mean, \
												std = self.std,
												shuffle=is_train)

		# convert to tensor for future calculations
		self.mean = torch.from_numpy(self.mean).to(self.device)
		self.std = torch.from_numpy(self.std).to(self.device)
		self.x_mean = torch.from_numpy(self.x_mean).to(self.device)
		self.x_std = torch.from_numpy(self.x_std).to(self.device).view(1, 1, motion_constants.NUM_JOINTS, 3)


	def build_network(self):
		# input_seq_dim = self.dataloader[self.mode].dataset.input_seq_dim
		# tgt_seq_dim = self.dataloader[self.mode].dataset.tgt_seq_dim

		# model_type = self.config['model']

		logging.info(f"Loading model...")

		if self.mode == "custom":
			offset = 2 if self.norm else 0
			data_dict = {}
			data_dict['input_dim'] = 137 + offset
			data_dict['output_dim'] = 137 + offset
		else:
			data_dict = self.dataloader[self.mode].dataset.get_data_dict()

		model_dict = self.config['model']
		
		# print(f"current function is build_network")
		# embed()
	
		self.model = load_model(data_config=data_dict, model_config=model_dict)
		self.model = self.model.to(self.device)
		self.model.zero_grad()
		
		self.criterion = nn.L1Loss()

		if self.pretrain:
			self.model.load_state_dict(torch.load(os.path.join(self.model_dir, 'model.pkl')))
			logging.info("pretrained model loaded")


	def build_optimizer(self):
		logging.info("Preparing optimizer...")

		self.optimizer = dadaptation.DAdaptAdam(self.model.parameters(), lr=1.0, decouple=True, weight_decay=1.0) # use AdamW
		self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=self.config['train']['lr_decay'])

		if self.pretrain:
			self.optimizer.load_state_dict(torch.load(os.path.join(self.model_dir + 'optimizer.pkl')))
			logging.info("optimizer loaded")


	def run(self, mode="test"):
		logging.info(f"Testing model with mode: {mode} ...")
		# if mode == "test":

		self.teacher_forcing_ratio = 0
		self.model.eval()

		render_data = []

		# for losses (in test and validation mode)
		epoch_loss = 0
		steps_per_epoch = len(self.dataloader[mode])
		# self.train_epoch = 0

		# data recording (in test mode)
		select_idx = 0
		if mode == "test":
			batch = self.config['test']['batch_size']
			select_idx =  random.randint(0, batch-1)
			print(f"selected index: {select_idx}")

		prev_input_seq = None
		for iterations, sampled_batch in enumerate(tqdm(self.dataloader[mode])):
			self.iterations = iterations
			with torch.no_grad():
				
				input_seq = sampled_batch['input_seq'].to(self.device)
				output_seq = self.model(input_seq) # ee, contact, output

				if self.norm:
					results = self.get_loss_norm(output_seq=output_seq, gt_tuple=sampled_batch, \
											get_results=(mode == "test"), \
											get_loss=True)
				else: 
					results = self.get_loss(output_seq=output_seq, gt_tuple=sampled_batch, \
											get_results=(mode == "test"), \
											get_loss=True)

				if results is not None:
					output_root, output_joint_rot = results
					tgt_root = sampled_batch['global_p'][...,0,:]
					tgt_rotations = sampled_batch['local_rot']

					rd = AERenderData(gt_root=tgt_root[select_idx], \
					gt_rot=tgt_rotations[select_idx].detach().cpu().numpy(), \
					output_root=output_root[select_idx].detach().cpu().numpy(),\
					output_rot=output_joint_rot[select_idx].detach().cpu().numpy())

					# TODO replace to root start - check working
					start_T = sampled_batch['root_start']
					rd.convert_to_matrix(start_T=start_T[select_idx])
					# rd.convert_to_matrix(start_T=None)

					render_data.append(rd)

			if mode in ["test", "validation"]:
				if self.loss_total.item() > 1000:
					print(f"current epoch loss: {epoch_loss} loss: {self.loss_total.item()} step: {iterations}")
					embed()
					continue

				epoch_loss += self.loss_total.item()
				prev_input_seq = input_seq
				# 	embed()

		if mode in ["test", "validation"]:
			epoch_loss /= steps_per_epoch
			logging.info(
				f"Test mode: {mode} | "
				f"{mode} loss: {epoch_loss}"
			)

		if mode == "test":
			ae_write_result_pkl(render_data_list=render_data, save_dir=os.path.join(self.directory, f"testset_{select_idx}/"))


	
	def train(self):
		self.writer = SummaryWriter(self.log_dir)
		logging.info("Training model...")
		torch.autograd.set_detect_anomaly(True)

		self.loss_total_min = 100000
		self.train_epoch = 0
		for epoch in range(self.config['train']['num_epoch']):
			self.train_epoch = epoch
			epoch_loss = 0
			self.model.train()
			self.teacher_forcing_ratio = 0.0 if (self.pretrain or (epoch-10) >= change_mode_epoch) else 1.0-((epoch-10)/change_mode_epoch)
			if epoch < 10:
				self.teacher_forcing_ratio = 1
			logging.info(
			f"Running epoch {epoch} | "
			f"teacher_forcing_ratio={self.teacher_forcing_ratio}"
			)

			steps_per_epoch = len(self.dataloader['train'])
			for iterations, sampled_batch in enumerate(self.dataloader['train']):
				self.iterations = iterations
				input_seq = sampled_batch['input_seq'].to(self.device)
				
				# add noise
				input_seq = input_seq + 0.02 * torch.randn(input_seq.shape).to(self.device)
				output_seq = self.model(input_seq) # hand (mid), foot, final_output (body)

				# tgt_seq = sampled_batch['tgt_seq'].to(self.device) # [batch, seq_len, dim]
				# global_pos = sampled_batch['global_p'].to(self.device)
				# root = sampled_batch['root'].to(self.device)

				if self.norm:
					results = self.get_loss_norm(output_seq=output_seq, gt_tuple=sampled_batch, get_results=False, get_loss=True)
				else: 
					results = self.get_loss(output_seq=output_seq, gt_tuple=sampled_batch, get_results=False, get_loss=True)

				if self.loss_total.item() > 1000:
					embed()
				self.optimize()
				if iterations % 5 == 0:
					self.update(epoch, steps_per_epoch, iterations)
				epoch_loss += self.loss_total.item()
			
			epoch_loss /= steps_per_epoch
			self.run(mode="validation") 
			self.save(epoch_loss, epoch)

	def get_loss_norm(self, output_seq, gt_tuple, get_results=False, get_loss=True):

		batch, seq_len, _ = output_seq.shape
		tgt_seq = gt_tuple['tgt_seq'].to(self.device) # [batch, seq_len, dim]		
		global_pos = gt_tuple['global_p'].to(self.device)
		tgt_root = gt_tuple['root'].to(self.device) # TODO check dim
		# gt_contact_label = gt_tuple['contact_label'].to(self.device)
		
		# normalize root (provide answer root pos by teacher forcing ration)
		# output_seq = tgt_seq.clone()
		output_root_height = output_seq[...,2:2+1]
		output_root_dir_vel = output_seq[...,2+1:2+2]
		output_root_pos_vel = output_seq[...,2+2:2+5]
		output_joint_rot = output_seq[...,2+5:]

		up_indice = 1 if motion_constants.UP_AXIS == "y" else 2
		axis = np.zeros(3)
		axis[up_indice] = 1.0
		axis_up = torch.from_numpy(np.tile(axis, (batch,1))).unsqueeze(1).float().to(self.device)
		facing_dir_vel_aa = axis_up * output_root_dir_vel 

		start = np.array([
			[1,0,0],
			[0,0,-1],
			[0,1,0],
		]).astype(dtype=np.float32)

		rotmat_id = torch.from_numpy(start[np.newaxis,...]).repeat(batch,1,1).to(self.device)
		output_root = []
		for i in range(seq_len):
			if i == 0:
				# facing_rotmat = normalized_local_T[:,i,0,:3,:3].clone()
				facing_rotmat = rotmat_id
				# start_pos = axis_up.squeeze(-2) * tgt_root[:,i,up_indice].clone()
				start_pos = axis_up.squeeze(1) * tgt_root[:,i,:].clone()
			else:
				rot_diff_mat = transforms.axis_angle_to_matrix(facing_dir_vel_aa[:,i])
				facing_rotmat = rot_diff_mat @ facing_rotmat
				pos_offset = (facing_rotmat.swapaxes(-2,-1) @ output_root_pos_vel[:,i,:].unsqueeze(-1)).squeeze(-1)
				if motion_constants.UP_AXIS == "z":
					start_pos[:,0] += pos_offset[:,0]
					start_pos[:,1] += -1 *pos_offset[:,2]
					start_pos[:,2] = (output_root_height[:,i,0]).clone()
				else: # y
					start_pos += pos_offset
					start_pos[:,1] = (output_root_height[:,i,0]).clone()

			output_root.append(start_pos.clone())
		output_root = torch.stack(output_root, dim=1) # TODO check 

		# embed()
		# get rot
		target_joint_rot = tgt_seq[...,2+5:].reshape(batch, seq_len, -1, 6)
		output_joint_rot = output_joint_rot.reshape(batch, seq_len, -1, 6)

		# mix
		idx_teacher = []
		idx_else = []
		for i in range(seq_len):
			if random.random() < self.teacher_forcing_ratio:
				idx_teacher.append(i)
			else:
				idx_else.append(i)

		root_view = tgt_root.reshape(batch, seq_len, 3)
		root_mix = output_root.clone()
		output_joint_rot_root_mix = output_joint_rot.clone()

		if len(idx_teacher) > 0:
			root_mix[:,idx_teacher,:] = root_view[:,idx_teacher,:]		
			output_joint_rot_root_mix[:,idx_teacher, 0, ...] = target_joint_rot[:, idx_teacher, 0, ...]

		output_joint_rot_root_mix = transforms.rotation_6d_to_matrix(output_joint_rot_root_mix)

		if not get_loss:
			return (output_root, transforms.rotation_6d_to_matrix(output_joint_rot)) if get_results else None

		# compare pos & ee 
		if self.skel_offset.shape[0] != batch:
			output_pos_mat = rot_matrix_fk_tensor(output_joint_rot_root_mix, root_mix, self.skel_offset[0:batch], self.skel_parent)
		else:
			output_pos_mat = rot_matrix_fk_tensor(output_joint_rot_root_mix, root_mix, self.skel_offset, self.skel_parent)

		root_diff = torch.abs(output_root - tgt_root) / self.x_std[...,0,:]

		# add root vel diff
		tgt_root_vel = tgt_root[1:] - tgt_root[:-1]
		output_root_vel = output_root[1:] - output_root[:-1]
		root_vel_diff = torch.mean(torch.abs(tgt_root_vel - output_root_vel))

		# add root rot
		root_rot_diff = self.criterion(output_seq[...,2+5:2+11], tgt_seq[...,2+5:2+11])
		self.root_mean_loss = torch.mean(root_diff) + root_vel_diff + root_rot_diff * 0.5 + torch.mean(root_diff[...,up_indice]) * 0.3 # height
		self.rotation_mse_loss = self.criterion(output_seq[...,2+5:], tgt_seq[...,2+5:])					

		# pos related loss
		pos_diff = torch.abs(global_pos - output_pos_mat) / self.x_std
		ee_diff = pos_diff[...,self.ee_idx+self.leg_idx,:]
		
		self.pos_mean_loss = torch.mean(pos_diff)
		self.ee_mean_loss = torch.mean(ee_diff)
		
		self.loss_total = self.config['train']['loss_pos_weight'] * self.pos_mean_loss + \
						self.config['train']['loss_ee_weight'] * self.ee_mean_loss + \
						self.config['train']['loss_quat_weight'] * self.rotation_mse_loss + \
						self.config['train']['loss_root_weight'] * self.root_mean_loss 
		
		return (output_root, transforms.rotation_6d_to_matrix(output_joint_rot)) if get_results else None


	def get_loss(self, output_seq, gt_tuple, get_results=False, get_loss=True):
		# mid_output, foot_output, output_seq = output_tuple
		# mid_ee, contact_output, output_seq = output_tuple

		batch, seq_len, _ = output_seq.shape
		# mid_seq = gt_tuple['mid_seq'].to(self.device)
		tgt_seq = gt_tuple['tgt_seq'].to(self.device) # [batch, seq_len, dim]

		# denormalize
		# output_seq = denormalize(output_seq, self.mean, self.std, self.device)
		# tgt_seq = denormalize(tgt_seq, self.mean, self.std, self.device)
		
		global_pos = gt_tuple['global_p'].to(self.device)
		root = gt_tuple['root'].to(self.device)
		gt_contact_label = gt_tuple['contact_label'].to(self.device)
		
		# normalize root (provide answer root pos by teacher forcing ration)
		output_root = output_seq[...,2:2+3] # TODO fix for frame-wise normalization
		
		output_joint_rot = output_seq[...,2+3:]
		output_joint_rot = output_joint_rot.reshape(batch, seq_len, -1, 6)
		target_joint_rot = tgt_seq[...,2+3:].reshape(batch, seq_len, -1, 6)

		# mix
		idx_teacher = []
		idx_else = []
		for i in range(seq_len):
			if random.random() < self.teacher_forcing_ratio:
				idx_teacher.append(i)
			else:
				idx_else.append(i)

		root_view = root.reshape(batch, seq_len, 3)
		root_mix = output_root.clone()
		output_joint_rot_root_mix = output_joint_rot.clone()

		if len(idx_teacher) > 0:
			root_mix[:,idx_teacher,:] = root_view[:,idx_teacher,:]		
			output_joint_rot_root_mix[:,idx_teacher, 0, ...] = target_joint_rot[:, idx_teacher, 0, ...]

		output_joint_rot_root_mix = transforms.rotation_6d_to_matrix(output_joint_rot_root_mix)

		if not get_loss:
			return (output_root, transforms.rotation_6d_to_matrix(output_joint_rot)) if get_results else None

		# compare pos & ee 
		if self.skel_offset.shape[0] != batch:
			output_pos_mat = rot_matrix_fk_tensor(output_joint_rot_root_mix, root_mix, self.skel_offset[0:batch], self.skel_parent)
		else:
			output_pos_mat = rot_matrix_fk_tensor(output_joint_rot_root_mix, root_mix, self.skel_offset, self.skel_parent)

		root_diff = torch.abs(output_seq[...,2:2+3] - tgt_seq[...,2:2+3]) / self.x_std[...,0,:]

		# add root rot
		root_rot_diff = self.criterion(output_seq[...,2+3:2+9], tgt_seq[...,2+3:2+9])
		self.root_mean_loss = torch.mean(root_diff) + root_rot_diff * 0.5 + torch.mean(root_diff[...,2]) * 0.3 # height
		self.rotation_mse_loss = self.criterion(output_seq[...,2+3:], tgt_seq[...,2+3:])					

		# pos related loss
		pos_diff = torch.abs(global_pos - output_pos_mat) / self.x_std
		ee_diff = pos_diff[...,self.ee_idx+self.leg_idx,:]
		
		self.pos_mean_loss = torch.mean(pos_diff)
		self.ee_mean_loss = torch.mean(ee_diff)
		
		self.loss_total = self.config['train']['loss_pos_weight'] * self.pos_mean_loss + \
						self.config['train']['loss_ee_weight'] * self.ee_mean_loss + \
						self.config['train']['loss_quat_weight'] * self.rotation_mse_loss + \
						self.config['train']['loss_root_weight'] * self.root_mean_loss 
		
		# if self.foot_vel_loss is not None:
		# 	self.loss_total += self.config['train']['loss_vel_weight'] * self.foot_vel_loss
		# print(f"loss for this step: {self.loss_total.item()}")
		return (output_root, transforms.rotation_6d_to_matrix(output_joint_rot)) if get_results else None
		
	def optimize(self):
		self.optimizer.zero_grad()
		self.loss_total.backward()
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
		self.optimizer.step()
		self.scheduler.step()
	
	def update(self, epoch, steps_per_epoch, idx):
		self.writer.add_scalar('loss_pos', self.pos_mean_loss.item(), global_step = epoch * steps_per_epoch + idx)
		self.writer.add_scalar('loss_ee', self.ee_mean_loss.item(), global_step = epoch * steps_per_epoch + idx)
		# self.writer.add_scalar('loss_foot', self.foot_pos_loss.item(), global_step = epoch * steps_per_epoch + idx)
		# self.writer.add_scalar('loss_mid', self.mid_mean_loss.item(), global_step = epoch * steps_per_epoch + idx)
		self.writer.add_scalar('loss_quat', self.rotation_mse_loss.item(), global_step = epoch * steps_per_epoch + idx)
		self.writer.add_scalar('loss_root', self.root_mean_loss.item(), global_step = epoch * steps_per_epoch + idx)
		# self.writer.add_scalar('loss_est', self.est_loss.item(), global_step = epoch * steps_per_epoch + idx)
		# self.writer.add_scalar('loss_contact', self.contact_loss.item(), global_step = epoch * steps_per_epoch + idx)

		# self.writer.add_scalar('loss_p_vel', self.vel_mean_loss.item(), global_step = epoch * steps_per_epoch + idx)
		# self.writer.add_scalar('loss_contact', self.contact_mse_loss.item(), global_step = epoch * steps_per_epoch + idx)
		self.writer.add_scalar('loss_total', self.loss_total.item(), global_step = epoch * steps_per_epoch + idx)
		# if self.foot_vel_loss is not None:
		# 	self.writer.add_scalar('loss_foot_vel', self.foot_vel_loss.item(), global_step = epoch * steps_per_epoch + idx)

	def save(self, epoch_loss, epoch):
		if (epoch % self.save_frequency == self.save_frequency-1) or epoch_loss < self.loss_total_min:
			# print("save")
			logging.info("Saving model")
			torch.save(self.model.state_dict(), self.model_dir + 'model.pkl')
			torch.save(self.optimizer.state_dict(), self.model_dir + 'optimizer.pkl')
			if epoch_loss < self.loss_total_min:
				self.loss_total_min = epoch_loss
		
		logging.info(f"Current Epoch: {epoch} | "
					 f"Current Loss: {epoch_loss} | "
					 f"Best Loss: {self.loss_total_min}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--test_name", type=str, default="")
	parser.add_argument("--mode", type=str, default="", choices=["test", "train", "custom"])
	parser.add_argument("--config", type=str, default="")

	# for inference of custom data
	parser.add_argument("--custom_datapath", type=str, default="")
	parser.add_argument("--window-size", type=int, help="The # of frames per window (0: follow training, -1: whole)", default=0)
	parser.add_argument("--offset-size", type=int, help="# of frames to slide per window", default=0)
	parser.add_argument("--custom-beta", action='store_true', help="Use custom beta")
	# parser.add_argument('--no-parse', action='store_true', help="Disable parsing")


	args = parser.parse_args()
	if not os.path.exists("./output/"):
		os.mkdir("./output/")
	
	autoencoder = AutoEncoder(args=args)

	if args.mode == "train":
		autoencoder.train()
	else:
		autoencoder.run(mode=args.mode)

	# autoencoder.init_train(directory=directory, config=args.config, pretrain=False)
	# autoencoder.train()

	# autoencoder.init_test(directory=directory)
	# autoencoder.run()
