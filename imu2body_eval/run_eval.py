import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
import argparse
from copy import deepcopy
import logging
import numpy as np
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
from IPython import embed 
import yaml
from functions import *
from pytorch3d import transforms
import dadaptation
from fairmotion.data import bvh
import imu2body_eval.amass_smplh as amass_smplh
from tqdm import tqdm
from fairmotion.ops import conversions
from imu2body.visualize_testset import RenderData 
import imu2body_eval.model
import constants.motion_data as motion_constants
from eval.metrics import * 
import glob 
import constants.path as path_constants
from fairmotion.utils import utils

from tensorboardX import SummaryWriter
 
logging.basicConfig(
	format="[%(asctime)s] %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	level=logging.INFO,
)

def set_seeds():
	torch.manual_seed(1234)
	np.random.seed(1234)
	random.seed(1234)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


class IMU2BodyNetworkEval(object):
	def __init__(self, args):
		set_seeds()

		# init 
		directory = "./output/" + args.test_name + "/"
		if not os.path.exists(directory):
			os.mkdir(directory)

		self.directory = directory
		self.mode = args.mode
		
		self.eval_test_directory = args.eval_path

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

		# save and vis related
		self.load_vis = args.load_vis
		# self.save_dir =  args.save_path
		self.save_dir = os.path.join(path_constants.BASE_PATH, args.save_path)
		utils.create_dir_if_absent(self.save_dir)


	def set_info(self, pretrain=False):
		is_train = True if self.mode == "train" else False
		self.pretrain = pretrain if is_train else True
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		logging.info(f"Using device: {self.device}")
		
		# set information from config and args
		self.save_frequency = self.config['train']['save_frequency']
		self.eval_frequency = None
		if 'eval' in self.config:
			self.eval_frequency = self.config['eval']['eval_frequency']
			self.eval_metric = ['mpjpe', 'root_mpjpe', 'mpjve', 'rootpe', 'pred_jitter', 'gt_jitter']

		self.set_skel_info()
		
		self.model_dir = os.path.join(self.directory, "model/")

		if not os.path.exists(self.model_dir):
			os.mkdir(self.model_dir)



	def set_skel_info(self):

		body_model = amass_smplh.load_body_model(motion_constants.SMPLH_BM_PATH)
		fairmotion_skel, _ = amass_smplh.create_skeleton_from_amass_bodymodel(bm=body_model)

		self.skel = fairmotion_skel
		self.skel_offset = fairmotion_skel.get_joint_offset_list()
		self.skel_parent = fairmotion_skel.get_parent_index_list()
		self.ee_idx = fairmotion_skel.get_index_joint(motion_constants.EE_JOINTS)
		self.foot_idx = fairmotion_skel.get_index_joint(motion_constants.FOOT_JOINTS) + fairmotion_skel.get_index_joint(motion_constants.toe_joints)
		self.hand_idx = fairmotion_skel.get_index_joint(motion_constants.HAND_JOINTS)
		self.leg_idx = fairmotion_skel.get_index_joint(motion_constants.LEG_JOINTS)

		self.mid_ee_idx = self.hand_idx + fairmotion_skel.get_index_joint(motion_constants.FOOT_JOINTS)
		self.skel_offset = torch.from_numpy(self.skel_offset[np.newaxis, np.newaxis, ...]).to(self.device).float() 		# expand skel offset into tensor

		if self.mode == "train":
			self.skel_offset = self.skel_offset.repeat(self.config['train']['batch_size'], motion_constants.preprocess_window, 1, 1)
	

	def load_data(self):

		data = np.load(self.directory + "mean_and_std.npz")
		self.mean = data['mean']
		self.std = data['std']

		x_data = np.load(self.directory + "x_mean_and_std.npz")
		self.x_mean = x_data['mean']
		self.x_std = x_data['std']

		# convert to tensor for future calculations
		self.mean = torch.from_numpy(self.mean).to(self.device).float()
		self.std = torch.from_numpy(self.std).to(self.device).float()
		self.x_mean = torch.from_numpy(self.x_mean).to(self.device).float()
		self.x_std = torch.from_numpy(self.x_std).to(self.device).view(1, 1, motion_constants.NUM_JOINTS, 3).float()

		self.eval_files = glob.glob(os.path.join(self.eval_test_directory, "*/*.pkl"))


	def build_network(self):
		logging.info(f"Loading model...")
		self.model_dir = os.path.join(self.directory, "model/")

		data_dict = {}
		data_dict['input_dim'] = 31
		data_dict['mid_dim'] = 12
		data_dict['output_dim'] = 135

		model_dict = self.config['model']

		self.model = imu2body_eval.model.load_model(data_config=data_dict, model_config=model_dict)
		self.model = self.model.to(self.device)
		self.model = nn.DataParallel(self.model)

		self.model.zero_grad()
		
		self.model.load_state_dict(torch.load(os.path.join(self.model_dir, 'model.pkl')))
		logging.info("pretrained model loaded")


	def build_optimizer(self):
		logging.info("Preparing optimizer...")

		self.optimizer = dadaptation.DAdaptAdam(self.model.parameters(), lr=1.0, decouple=True, weight_decay=1.0) # use AdamW
		self.optimizer.load_state_dict(torch.load(os.path.join(self.model_dir + 'optimizer.pkl')))
		logging.info("optimizer loaded")


	def run_per_file(self, file_dict):
		sampled_batch = file_dict

		# create placeholder for pred pos, pred rot, gt pos and gt rot
		total_length = sampled_batch['total_length']
		predicted_position = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3))
		predicted_rot = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3, 3))
		gt_position = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3))
		gt_rot = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3, 3))

		input_seq = sampled_batch['input_seq'].to(self.device)
		input_seq = (input_seq - self.mean) / self.std 
		output_tuple = self.model(input_seq)
		results = self.get_loss(output_tuple=output_tuple, gt_tuple=sampled_batch, \
								get_results=False, \
								get_loss=True, \
								is_eval=True)
		
		start_T = sampled_batch['head_start'].to(self.device)

		# get pred into world coord
		pred_pos_to_world = start_T[...,:3,:3].to(self.device) @ results['pred_pos'].unsqueeze(-1)
		pred_pos_to_world = pred_pos_to_world[...,0] + start_T[...,:3,3]
		pred_rotmat = transforms.rotation_6d_to_matrix(results['pred_rot'])
		pred_rotmat[...,0:1,:,:] = start_T[...,:3,:3] @ pred_rotmat[...,0:1,:,:]

		# get gt into world coord
		gt_pos_to_world = start_T[...,:3,:3].to(self.device) @ results['gt_pos'].unsqueeze(-1)
		gt_pos_to_world = gt_pos_to_world[...,0] + start_T[...,:3,3]
		gt_rotmat = transforms.rotation_6d_to_matrix(results['gt_rot'])
		gt_rotmat[...,0:1,:,:] = start_T[...,:3,:3] @ gt_rotmat[...,0:1,:,:]

		# into single seq
		batch, seq_len, J, _ = pred_pos_to_world.shape
		for idx, start_frame in enumerate(sampled_batch['start_frame']):
			predicted_position[start_frame:start_frame+seq_len] = pred_pos_to_world[idx]
			predicted_rot[start_frame:start_frame+seq_len] = pred_rotmat[idx]
			gt_position[start_frame:start_frame+seq_len] = gt_pos_to_world[idx]
			gt_rot[start_frame:start_frame+seq_len] = gt_rotmat[idx]

		predicted_angle_np = conversions.R2A(predicted_rot.cpu().numpy())
		predicted_angle = torch.from_numpy(predicted_angle_np).cuda().float()
		predicted_root_angle = predicted_angle[...,0,:] 

		gt_angle_np = conversions.R2A(gt_rot.cpu().numpy())
		gt_angle = torch.from_numpy(gt_angle_np).cuda().float()
		gt_root_angle = gt_angle[...,0,:] 
				

		# after running iterations get numbers
		upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
		lower_index = [0, 1, 2, 4, 5, 7, 8] # 10,11 is not considered in imus. (why? TIP does not have ankle joints)
		hand_index = [20, 21]
		foot_index = [7, 8]
		eval_log = {}
		for metric in self.eval_metric:
			eval_metric = get_metric_function(metric)(
					predicted_position,
					predicted_angle,
					predicted_root_angle,
					gt_position,
					gt_angle,
					gt_root_angle,
					upper_index,
					lower_index,
					hand_index,
					foot_index,
					fps=motion_constants.FPS,
					root_rel=True
				).cpu().numpy()
			eval_log[metric] = eval_metric 
		
		# add filename
		parts = sampled_batch['filename'].split('/')
		filename = '/'.join(parts[3:])
		eval_log['filename'] = filename
		torch.cuda.empty_cache()

		# convert to fairmotion for visualization -> comment out for running eval (num) makes it slow
		# if self.save_dir != "" or self.load_vis:
		# 	num_frames = total_length
		# 	gt_T = np.zeros(shape=(num_frames, 22, 4, 4))
		# 	output_T = np.zeros(shape=(num_frames, 22, 4, 4))

		# 	gt_T[:,0,:3,3] = gt_position[:,0,:].clone().cpu().numpy()
		# 	gt_T[:,:,:3,:3] = gt_rot.clone().cpu().numpy()

		# 	output_T[:,0,:3,3] = predicted_position[:,0,:].clone().cpu().numpy()
		# 	output_T[:,:,:3,:3] = predicted_rot.clone().cpu().numpy()
		# 	# save results
		# 	# save_filename = filename.replace(".npz", "").replace("_poses","").replace("/", "_")
		# 	# save_filepath = os.path.join(self.save_dir, f"{save_filename}.pkl")

		# 	# convert into fairmotion
		# 	gt_motion = motion_classes.Motion.from_matrix(gt_T, skel=deepcopy(self.skel))
		# 	output_motion = motion_classes.Motion.from_matrix(output_T, skel=deepcopy(self.skel))

		# 	# dict
		# 	save_dict = {}
		# 	save_dict['gt_motion'] = gt_motion
		# 	save_dict['output_motion'] = output_motion

		# 	# with open(save_filepath, "wb") as file:
		# 	# 	pickle.dump(save_dict, file)

		# if self.load_vis:
		# 	eval_log['motion'] = gt_motion, output_motion

		return eval_log

	def eval(self):
		
		logging.info(f"Eval with testset ...")
		self.teacher_forcing_ratio = 0
		self.model.eval()

		self.eval_log = {}
		for metric in self.eval_metric:
			self.eval_log[metric] = []
		
		count = 0
		filenames = []
		self.eval_log_by_filename = {}

		render_result_dict = {}
		render_result_dict['fps'] = 30.0
		render_result_dict['seq_len'] = []
		render_result_dict['idx'] = []
		render_result_dict['motion'] = []

		for filepath in tqdm(self.eval_files):
			with open(filepath, "rb") as file:
				file_dict = pickle.load(file)
			eval_log_per_file = self.run_per_file(file_dict=file_dict)
			if self.load_vis:
				render_result_dict['motion'].append(eval_log_per_file['motion'])
				render_result_dict['seq_len'].append(eval_log_per_file['motion'][0].num_frames())
				render_result_dict['idx'].append(filepath)
			if eval_log_per_file['filename'] in self.eval_log_by_filename:
				embed()
			self.eval_log_by_filename[eval_log_per_file['filename']] = eval_log_per_file
			for metric in self.eval_metric:
				self.eval_log[metric].append(eval_log_per_file[metric])
		
		print(f"Done.")
		logging.info(f"-----------------------EVAL RESULT-----------------------------------------------")
		for metric in self.eval_metric:
			if 'jitter' in metric:
				continue
			print(f"metric: {metric} value: {np.mean(np.array(self.eval_log[metric])) * metrics_coeffs[metric]:.2f}")
		print(f"metric: jitter value: {np.mean(np.array(self.eval_log['pred_jitter'])) / np.mean(np.array(self.eval_log['gt_jitter'])):.2f}")
		logging.info(f"----------------------------------------------------------------------------------")

		if self.load_vis:
			return render_result_dict
		


	def get_loss(self, output_tuple, gt_tuple, get_results=False, get_loss=True, is_eval=False):
		mid_ee, contact_output, output_seq = output_tuple

		batch, seq_len, _ = output_seq.shape

		mid_seq = gt_tuple['mid_seq'].to(self.device)
		tgt_seq = gt_tuple['tgt_seq'].to(self.device) # [batch, seq_len, dim]
		global_pos = gt_tuple['global_p'].to(self.device)
		root = gt_tuple['root'].to(self.device)
		gt_contact_label = gt_tuple['contact_label'].to(self.device)

		output_root = output_seq[...,:3]
		output_joint_rot = output_seq[...,3:]
		output_joint_rot = output_joint_rot.reshape(batch, seq_len, -1, 6)
		target_joint_rot = tgt_seq[...,3:].reshape(batch, seq_len, -1, 6)

		output_joint_rotmat = transforms.rotation_6d_to_matrix(output_joint_rot)

		if not get_loss:
			return (output_root, transforms.rotation_6d_to_matrix(output_joint_rot)) if get_results else None

		# compare pos & ee 
		if self.skel_offset.shape[0] != batch:
			output_pos_mat = rot_matrix_fk_tensor(output_joint_rotmat, output_root, self.skel_offset[0:batch], self.skel_parent)
		else:
			output_pos_mat = rot_matrix_fk_tensor(output_joint_rotmat, output_root, self.skel_offset, self.skel_parent)

		if is_eval:
			result_dict = {}
			result_dict['pred_pos'] = output_pos_mat.clone().detach()
			result_dict['pred_rot'] = output_joint_rot.clone().detach()
			result_dict['gt_pos'] = global_pos.clone().detach()
			result_dict['gt_rot'] = target_joint_rot.clone().detach()

			return result_dict
		

		change_mode_epoch = 40
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

		# est loss 
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
	parser.add_argument("--eval-path", type=str, default="")

	parser.add_argument("--save-path", type=str, default="")
	parser.add_argument("--load-vis", action="store_true")

	args = parser.parse_args()
	if not os.path.exists("./output/"):
		os.mkdir("./output/")
	
	imu2body_network = IMU2BodyNetworkEval(args=args)

	render_result_dict = imu2body_network.eval()

	# if args.load_vis:
	# 	from visualizer.render_args import add_render_argparse
	# 	import visualizer.pip_visualizer as visualizer

	# 	parser = add_render_argparse(parser=parser)
	# 	args = parser.parse_args()
	# 	args.axis_up = motion_constants.UP_AXIS	

	# 	vis = visualizer.load_visualizer(result_dict=render_result_dict, args=args)

	# 	vis.run() 
