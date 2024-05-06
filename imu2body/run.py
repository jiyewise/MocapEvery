import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
import argparse
from copy import deepcopy
import logging
import numpy as np
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torch.nn as nn
import torch.optim as optim
from IPython import embed 
import yaml
from dataset import *
from functions import *
from pytorch3d import transforms
import dadaptation
from fairmotion.data import bvh
import amass
import imu2body_eval.amass_smplh as amass_smplh
from tqdm import tqdm
from fairmotion.ops import conversions
from imu2body.visualize_testset import RenderData 
import imu2body.model
import constants.motion_data as motion_constants
from eval.metrics import * 

from tensorboardX import SummaryWriter
 
logging.basicConfig(
	format="[%(asctime)s] %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	level=logging.INFO,
)

change_mode_epoch = 40

def set_seeds():
	torch.manual_seed(1234)
	np.random.seed(1234)
	random.seed(1234)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


smplx_bm_path = "../data/smpl_models/smplx/SMPLX_NEUTRAL.npz"
smplh_bm_path = "../data/smpl_models/smplh/male/model.npz"

CUR_BM_TYPE = "smplx"

class IMU2BodyNetwork(object):
	def __init__(self, args):
		set_seeds()

		# init 
		directory = "./output/" + args.test_name + "/"
		if not os.path.exists(directory):
			os.mkdir(directory)

		self.directory = directory
		self.mode = args.mode
		
		# open config file
		config_dir = "./config/"+args.config + ".yaml" if self.mode == "train" else self.directory + "config.yaml"
		self.config = yaml.safe_load(open(config_dir, 'r').read())

		if self.mode == "train":
			os.system('cp {} {}'.format('./config/'+args.config+'.yaml', directory+'config.yaml'))

		self.data_path = self.config['data']['preprocess']

		logging.info(f"Starting in {self.mode} mode...")

		self.set_info()
		self.load_data()  
		
		self.build_network()
		self.build_optimizer()


	def set_info(self, pretrain=False):
		is_train = True if self.mode == "train" else False
		self.pretrain = pretrain if is_train else True
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		logging.info(f"Using device: {self.device}")
		
		# set information from config and args
		self.save_frequency = self.config['train']['save_frequency']
		self.set_skel_info() # load skeleton info (this is needed for train and test)
		
		self.log_dir = os.path.join(self.directory, "log/")
		self.model_dir = os.path.join(self.directory, "model/")

		if not os.path.exists(self.log_dir):
			os.mkdir(self.log_dir)

		if not os.path.exists(self.model_dir):
			os.mkdir(self.model_dir)

		self.train_epoch = 0
		
		self.loss_func = self.get_loss

	def set_skel_info(self):
		
		if CUR_BM_TYPE == "smplx":
			body_model = amass.load_body_model(motion_constants.BM_PATH) 
			fairmotion_skel, _ = amass.create_skeleton_from_amass_bodymodel(bm=body_model)

		elif CUR_BM_TYPE == "smplh":
			body_model = amass_smplh.load_body_model(motion_constants.SMPLH_BM_PATH) 
			fairmotion_skel, _ = amass_smplh.create_skeleton_from_amass_bodymodel(bm=body_model)
		else:
			raise NotImplementedError("Only smplx and smplh are supported!")
		
		self.skel = fairmotion_skel
		self.skel_offset = fairmotion_skel.get_joint_offset_list()
		self.skel_parent = fairmotion_skel.get_parent_index_list()
		self.ee_idx = fairmotion_skel.get_index_joint(motion_constants.EE_JOINTS)
		self.foot_idx = fairmotion_skel.get_index_joint(motion_constants.FOOT_JOINTS) + fairmotion_skel.get_index_joint(motion_constants.toe_joints)
		self.hand_idx = fairmotion_skel.get_index_joint(motion_constants.HAND_JOINTS)
		# this is to solve overfitting on foot
		self.leg_idx = fairmotion_skel.get_index_joint(motion_constants.LEG_JOINTS)

		self.mid_ee_idx = self.hand_idx + fairmotion_skel.get_index_joint(motion_constants.FOOT_JOINTS)
		self.skel_offset = torch.from_numpy(self.skel_offset[np.newaxis, np.newaxis, ...]).to(self.device).float() 		# expand skel offset into tensor
		if self.mode == "train":
			self.skel_offset = self.skel_offset.repeat(self.config['train']['batch_size'], motion_constants.preprocess_window, 1, 1)
	

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

		for fname in fnames:
			if fname == "train":
				batch_size = self.config['train']['batch_size']
				train_fnames = [os.path.join(self.data_path, f"train_{i+1}.pkl") for i in range(5)] # TODO fix so automatically would be all read
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
				batch_size = self.config['train']['batch_size'] if fname == "validation" else self.config['test']['batch_size']
				self.dataloader[fname] = get_loader(dataset_path=os.path.join(self.data_path, f"{fname}.pkl"), \
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

		logging.info(f"Loading model...")

		data_dict = self.dataloader[self.mode].dataset.get_data_dict()
		model_dict = self.config['model']

		self.model = imu2body.model.load_model(data_config=data_dict, model_config=model_dict)
		self.model = self.model.to(self.device)
		self.model = nn.DataParallel(self.model)

		self.model.zero_grad()
		
		self.criterion = nn.L1Loss()

		self.contact_criterion = nn.BCEWithLogitsLoss()

		if self.pretrain:
			self.model.load_state_dict(torch.load(os.path.join(self.model_dir, 'model.pkl')))
			logging.info("pretrained model loaded")


	def build_optimizer(self):
		logging.info("Preparing optimizer...")

		self.optimizer = dadaptation.DAdaptAdam(self.model.parameters(), lr=1.0, decouple=True, weight_decay=1.0) # use AdamW
		# self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=self.config['train']['lr_decay'])

		if self.pretrain:
			self.optimizer.load_state_dict(torch.load(os.path.join(self.model_dir + 'optimizer.pkl')))
			logging.info("optimizer loaded")


	def run(self, mode="test"):
		logging.info(f"Testing model with mode: {mode} ...")

		self.teacher_forcing_ratio = 0
		self.model.eval()

		render_data = []

		# for losses (in test and validation mode)
		epoch_loss = 0
		steps_per_epoch = len(self.dataloader[mode])

		# data recording (in test mode)
		select_idx = 0
		if mode == "test":
			batch = self.config['test']['batch_size']
			select_idx =  random.randint(0, batch-1)
			print(f"selected index: {select_idx}")

		for iterations, sampled_batch in enumerate(tqdm(self.dataloader[mode])):
			with torch.no_grad():
				input_seq = sampled_batch['input_seq'].to(self.device)

				output_tuple = self.model(input_seq) # ee, contact, output

				results = self.loss_func(output_tuple=output_tuple, gt_tuple=sampled_batch, \
			    						get_results=(mode == "test"), \
										get_loss=True)

				if results is not None:
					output_root, output_joint_rot = results
					tgt_root = sampled_batch['global_p'][...,0,:]
					tgt_rotations = sampled_batch['local_rot']

					rd = RenderData(gt_root=tgt_root[select_idx], \
					gt_rot=tgt_rotations[select_idx].detach().cpu().numpy(), \
					output_root=output_root[select_idx].detach().cpu().numpy(),\
					output_rot=output_joint_rot[select_idx].detach().cpu().numpy())

					start_T = sampled_batch['head_start']
					rd.convert_to_matrix(start_T=start_T[select_idx])

					# reshape
					other_est = {}
					other_est['ee'] = output_tuple[0][select_idx].detach().cpu().numpy()

					render_data.append(rd)

			if mode in ["test", "validation"]:
				epoch_loss += self.loss_total.item()

		if mode in ["test", "validation"]:			
			epoch_loss /= steps_per_epoch
			logging.info(
				f"Test mode: {mode} | "
				f"{mode} loss: {epoch_loss}"
			)

		if mode == "test":
			write_result_pkl(render_data_list=render_data, save_dir=os.path.join(self.directory, f"testset_{select_idx}/"))

	
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
			self.teacher_forcing_ratio = 0.0 if (self.pretrain or epoch >= change_mode_epoch) else float((change_mode_epoch-epoch)/change_mode_epoch)

			logging.info(
			f"Running epoch {epoch} | "
			f"teacher_forcing_ratio={self.teacher_forcing_ratio}"
			)

			steps_per_epoch = len(self.dataloader['train'])
			for iterations, sampled_batch in enumerate(tqdm(self.dataloader['train'])):
				input_seq = sampled_batch['input_seq'].to(self.device)
				
				# add noise
				input_seq = input_seq + 0.01 * torch.randn(input_seq.shape).to(self.device)
				output_tuple = self.model(input_seq) # hand (mid), foot, final_output (body)

				results = self.loss_func(output_tuple=output_tuple, gt_tuple=sampled_batch, get_results=False, get_loss=True)
				
				self.optimize()
				if iterations % 5 == 0:
					self.update(epoch, steps_per_epoch, iterations)
				epoch_loss += self.loss_total.item()
			
			epoch_loss /= steps_per_epoch
			self.run(mode="validation") 
			self.save(epoch_loss, epoch)


	def get_loss(self, output_tuple, gt_tuple, get_results=False, get_loss=True, is_eval=False):
		# mid_output, foot_output, output_seq = output_tuple
		mid_ee, contact_output, output_seq = output_tuple

		batch, seq_len, _ = output_seq.shape

		mid_seq = gt_tuple['mid_seq'].to(self.device)
		tgt_seq = gt_tuple['tgt_seq'].to(self.device) # [batch, seq_len, dim]
		global_pos = gt_tuple['global_p'].to(self.device)
		root = gt_tuple['root'].to(self.device)
		gt_contact_label = gt_tuple['contact_label'].to(self.device)

		# normalize root (provide answer root pos by teacher forcing ration)
		output_root = output_seq[...,:3]
		output_joint_rot = output_seq[...,3:]
		output_joint_rot = output_joint_rot.reshape(batch, seq_len, -1, 6)
		target_joint_rot = tgt_seq[...,3:].reshape(batch, seq_len, -1, 6)

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

		if is_eval:
			result_dict = {}
			result_dict['pred_pos'] = output_pos_mat.clone()
			result_dict['pred_rot'] = output_joint_rot.clone()
			result_dict['gt_pos'] = global_pos.clone()
			result_dict['gt_rot'] = target_joint_rot.clone()

			return result_dict


		root_diff = torch.abs(output_seq[...,:3] - tgt_seq[...,:3]) / self.x_std[...,0,:]

		# add root rot
		root_rot_diff = self.criterion(output_seq[...,3:9], tgt_seq[...,3:9])
		self.root_mean_loss = torch.mean(root_diff) + root_rot_diff * 0.5
		self.rotation_mse_loss = self.criterion(output_seq[...,3:], tgt_seq[...,3:])					

		# pos related loss
		pos_diff = torch.abs(global_pos - output_pos_mat) / self.x_std
		ee_diff = pos_diff[...,self.ee_idx+self.leg_idx,:]
		foot_diff = pos_diff[..., self.foot_idx,:] # output of the final layer
		
		self.pos_mean_loss = torch.mean(pos_diff)
		self.ee_mean_loss = torch.mean(ee_diff)
		self.foot_pos_loss = torch.mean(foot_diff)
		
		self.foot_vel_loss = None
		if self.train_epoch > change_mode_epoch:
			vel = global_pos[...,1:,:,:] - global_pos[...,:-1,:,:]
			output_vel = output_pos_mat[...,1:,:,:] - output_pos_mat[...,:-1,:,:]
			vel_diff = torch.abs(output_vel - vel)
			vel_diff = torch.abs(output_vel - vel) / self.x_std
			self.foot_vel_loss = torch.mean(vel_diff[...,self.foot_idx, :])

		# mid ee 
		mid_ee_reshape = mid_ee.reshape(batch, seq_len, -1, 3)
		mid_seq_reshape = mid_seq.reshape(batch, seq_len, -1, 3)
		mid_pos_diff = torch.abs(mid_ee_reshape - mid_seq_reshape) / self.x_std[...,self.mid_ee_idx,:]
		self.mid_mean_loss = torch.mean(mid_pos_diff)

		mid_est_diff = torch.abs(output_pos_mat[...,self.mid_ee_idx,:] - mid_ee_reshape) / self.x_std[...,self.mid_ee_idx,:]
		self.est_loss = torch.mean(mid_est_diff)

		# contact classifier loss
		self.contact_loss = self.contact_criterion(contact_output, gt_contact_label)

		self.loss_total = self.config['train']['loss_pos_weight'] * self.pos_mean_loss + \
						self.config['train']['loss_foot_weight'] * self.foot_pos_loss + \
						self.config['train']['loss_ee_weight'] * self.ee_mean_loss + \
						self.config['train']['loss_mid_weight'] * self.mid_mean_loss + \
						self.config['train']['loss_quat_weight'] * self.rotation_mse_loss + \
						self.config['train']['loss_root_weight'] * self.root_mean_loss + \
						self.config['train']['loss_est_weight'] * self.est_loss + \
						self.config['train']['loss_contact_weight'] * self.contact_loss

		if self.foot_vel_loss is not None:
			self.loss_total += self.config['train']['loss_vel_weight'] * self.foot_vel_loss

		return (output_root, transforms.rotation_6d_to_matrix(output_joint_rot)) if get_results else None
		
	def optimize(self):
		self.optimizer.zero_grad()
		self.loss_total.backward()
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
		self.optimizer.step()
	
	
	def update(self, epoch, steps_per_epoch, idx):
		self.writer.add_scalar('loss_pos', self.pos_mean_loss.item(), global_step = epoch * steps_per_epoch + idx)
		self.writer.add_scalar('loss_ee', self.ee_mean_loss.item(), global_step = epoch * steps_per_epoch + idx)
		self.writer.add_scalar('loss_foot', self.foot_pos_loss.item(), global_step = epoch * steps_per_epoch + idx)
		self.writer.add_scalar('loss_mid', self.mid_mean_loss.item(), global_step = epoch * steps_per_epoch + idx)
		self.writer.add_scalar('loss_quat', self.rotation_mse_loss.item(), global_step = epoch * steps_per_epoch + idx)
		self.writer.add_scalar('loss_root', self.root_mean_loss.item(), global_step = epoch * steps_per_epoch + idx)
		self.writer.add_scalar('loss_est', self.est_loss.item(), global_step = epoch * steps_per_epoch + idx)
		self.writer.add_scalar('loss_contact', self.contact_loss.item(), global_step = epoch * steps_per_epoch + idx)

		self.writer.add_scalar('loss_total', self.loss_total.item(), global_step = epoch * steps_per_epoch + idx)
		if self.foot_vel_loss is not None:
			self.writer.add_scalar('loss_foot_vel', self.foot_vel_loss.item(), global_step = epoch * steps_per_epoch + idx)


	def save(self, epoch_loss, epoch):
		if (epoch % self.save_frequency == self.save_frequency-1) or epoch_loss < self.loss_total_min:
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


	args = parser.parse_args()
	if not os.path.exists("./output/"):
		os.mkdir("./output/")
	
	imu2body_network = IMU2BodyNetwork(args=args)

	if args.mode == "train":
		imu2body_network.train()
	else:
		imu2body_network.run(mode=args.mode)