import sys, os
import numpy as np
import pickle
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
import argparse
from copy import deepcopy
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
from IPython import embed 
# from bvh import *
import yaml
from imu2body.dataset import *
from imu2body.functions import *
from pytorch3d import transforms
# import dadaptation
from fairmotion.data import bvh
# import amass
from tqdm import tqdm
from fairmotion.ops import conversions
from imu2body.visualize_testset import RenderData 
import imu2body.model
import constants.motion_data as motion_constants
from interaction.contact import *
from realdata.realdata_collection import *
from imu2body.functions import * 
from fairmotion.utils import utils
import pickle 
import dadaptation
from realdata.footcleanup import *

# reading result related
translate_offset = np.array([0.3,0.0,0.0])
def get_info_cam_signal(cam_signal):
	cam_signal.load_pcd_result()
	return cam_signal

def parse_realdata_network_output(args, load_motion=True):
	# return dictionary
	result_dict = {}

	# first read signals
	cam_filepath = os.path.join(path_constants.DATA_PATH, f"{args.rdc_folder}/cam_signal.pkl")
	imu_filepath = os.path.join(path_constants.DATA_PATH, f"{args.rdc_folder}/imu_signal.pkl")

	if sys.version_info < (3, 8):
		import pickle5 as pickle 
	else:
		import pickle

	with open(cam_filepath, "rb") as cam_file:
		cam_data = pickle.load(cam_file)
	with open(imu_filepath, "rb") as imu_file:
		imu_data = pickle.load(imu_file)
	
	# result_dict['pcd'], result_dict['img_file_list'] = 
	camera_head_signal = get_info_cam_signal(cam_data['camera_signal'])
	result_dict['imu_signal'] = imu_data
	result_dict['head_signal'] = camera_head_signal
	result_dict['pcd'] = camera_head_signal.pcd 
	result_dict['cam_ext'] = camera_head_signal.mocap_cam

	# load cam intrinsics for reprojection 
	undistort_int_path = os.path.join(path_constants.DATA_PATH, f"{args.rdc_folder}/sfm/undistort_calib.txt")
	if os.path.exists(undistort_int_path):
		cam_int_list = np.loadtxt(undistort_int_path, delimiter=" ")
		fx, fy, cx, cy = cam_int_list
		K = np.eye(3)
		K[0,0] = fx
		K[0,2] = cx
		K[1,1] = fy
		K[1,2] = cy
		result_dict['cam_int'] = K
	else:
		print(f"undistorted camera intrinsics txt file not found!")
		result_dict['cam_int'] = None 

	bm_path = motion_constants.BM_PATH if args.bm_type == "smplx" else motion_constants.SMPLH_BM_PATH

	body_model = amass.load_body_model(bm_path=bm_path)

	amass_skel, offset = amass.create_skeleton_from_amass_bodymodel(bm=body_model, \
																	betas=None 
																)
	
	# this is for mesh rendering (for rendering demos)
	result_dict['bm_path'] = bm_path
	result_dict['skel_with_offset'] = amass_skel, offset

	# other information
	result_dict['fps'] = motion_constants.FPS

	if not load_motion:
		result_dict['bm_type'] = args.bm_type 
		result_dict['skel_w_offset'] = amass_skel, offset
		return result_dict
	
	# read network output from ../imu2body/
	output_type = ["watch_cam"]
	output_motion = {}
	output_contact_label = {}
	for i, otype in enumerate(output_type):
		if args.bm_type == "smplx":
			network_output_path = f"../imu2body/output/{args.test_name}/custom/{args.rdc_folder}.pkl"
		if args.bm_type == "smplh":
			network_output_path = f"../imu2body_eval/output/{args.test_name}/custom/{args.rdc_folder}.pkl"
		if not os.path.exists(network_output_path):
			print(f"{network_output_path} does not exist!")
			continue 
		
		output_data = pickle.load(open(network_output_path, "rb"))

		output_motion_list = utils.run_parallel(convert_motion, output_data['output'], num_cpus=20, skel=amass_skel)
		offset = 0

		num_motions = len(output_data['output'])
		for i in tqdm(range(1, num_motions)):
			if len(output_motion_list[i].poses) == 0:
				continue
			elif len(output_motion_list[i].poses) > 7:
				target_pose = output_motion_list[i].poses[-offset]
				for i_before in range(7, 0, -1):
					weight = i_before * (1/7.0)
					# print(f"i_before: {i_before}, weight: {weight}")
					interpolated_pose = motion_class.Pose.interpolate(deepcopy(output_motion_list[0].poses[-i_before]), target_pose, 1-weight)
					output_motion_list[0].poses[-i_before] = interpolated_pose

			output_motion_list[0].poses += output_motion_list[i].poses # output_motion_list[i].poses[-offset:]
		
		output_motion[otype] = output_motion_list[0]
		output_contact_label[otype] = np.concatenate(output_data['contact_label'])

		if 'contact' in output_data:
			result_dict['contact'] = output_data['contact']

	# foot cleanup
	if 'cleanup' in vars(args) and args.cleanup:
		fc_module = FootCleanup(motion=output_motion["watch_cam"], contact_label=output_contact_label["watch_cam"])
		fc_module.foot_cleanup()

	result_dict['motion'] = output_motion
	result_dict['motion_contact_label'] = output_contact_label

	# other information
	result_dict['fps'] = motion_constants.FPS
	result_dict['seq_len'] = len(output_motion_list[0].poses)
	result_dict['translate_offset'] = translate_offset

	return result_dict

# convert network output into motion data

class MotionData(object):
	def __init__(self, gt_root, gt_rot, output_root, output_rot, use_gt=True):
		self.output_root = output_root # [seq_len, 3]
		self.output_rot = output_rot # [seq_len, J, 4]
		if use_gt:
			self.gt_root = gt_root # [seq_len, 3]
			self.gt_rot = gt_rot # [seq_len, J, 4]
		self.start_T = None
		self.use_gt = use_gt
		self.start_frame = None
		self.parse_frame = None
		self.contact_label = None
		self.seq_len = self.output_root.shape[0]

	def convert_to_T(self, rot, root):
		if rot.shape[-1] == 3:
			return rotmat_to_T_motion(rot, root)
		else:
			assert rot.shape[-1] == 4, "rotations should be either rotation matrices or quaternions"
			return quat_to_T_motion(rot, root)
	
	def convert_to_matrix(self, scale=1, use_gt=True, start_T=None):
		self.output_root *= scale
		self.output_T = self.convert_to_T(self.output_rot, self.output_root)
		if start_T is not None:
			self.start_T = start_T
			self.output_T[...,0:1,:,:] = start_T @ self.output_T[...,0:1,:,:]	

			if use_gt:
				self.gt_root *= scale
				self.gt_T = self.convert_to_T(self.gt_rot, self.gt_root)
				self.gt_T[...,0:1,:,:] = start_T @ self.gt_T[...,0:1,:,:]

	
	def set_frame(self, start_frame=None, parse_frame=None):
		self.start_frame = start_frame
		self.parse_frame = parse_frame

	def convert_to_logmap(self, scale=0.01, use_gt=False):
		self.output_root *= scale 
		aa = conversions.R2A(self.output_rot)
		frame, joint, _ = aa.shape
		aa = aa.reshape(frame, -1)
		self.logmap = np.concatenate((self.output_root, aa), axis=-1, dtype=np.float32)
	
	def set_contact_label(self, contact_label):
		self.contact_label = contact_label 


def write_motion_data_result(motion_data_list, save_dir, filename="", custom_config=None, contact_manager=None):
	data = {}
	data['output'] = []
	data['gt'] = []
	data['contact_label'] = []

	for rd in motion_data_list:
		start_frame = 0 if rd.start_frame is None else rd.start_frame
		parse_frame = rd.output_T.shape[0] if rd.parse_frame is None else rd.parse_frame
		data['output'].append(rd.output_T[start_frame:parse_frame, ...])
		data['contact_label'].append(rd.contact_label[start_frame:parse_frame, ...])
		if rd.use_gt:
			data['gt'].append(rd.gt_T)
		else:
			data['gt'].append(None)
	
	# save config also
	if custom_config is not None:
		data.update(custom_config)
	
	# save contact manager
	if contact_manager is not None:
		data['contact'] = contact_manager.contact_dict

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	
	pkl_filename = os.path.join(save_dir, f"{filename}.pkl")
	folder_path = os.path.dirname(pkl_filename)
	utils.create_dir_if_absent(folder_path)
	
	with open(pkl_filename, 'wb') as file:
		pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


# running network related 
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


class IMU2BodyRealdataRun(object):
	def __init__(self, args):
		set_seeds()

		# init 
		directory = "../imu2body/output/" + args.test_name + "/"
		if args.bm_type == "smplh":
			directory = "../imu2body_eval/output/" + args.test_name + "/"
		self.directory = directory
		
		# open config file
		config_dir = self.directory + "config.yaml"
		self.config = yaml.safe_load(open(config_dir, 'r').read())

		# open realdata collection and align
		self.rdc = RealDataCollection2(args.rdc_folder)
		self.rdc.load()
		self.rdc.save_signal_info()

		self.config['custom'] = {} # add custom related config in self.config
		self.config['custom']['seq_len'] = motion_constants.preprocess_window
		self.config['custom']['offset'] = 25
		self.is_realdata = True

		self.custom_filename = f"custom/{args.rdc_folder}"

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		logging.info(f"Using device: {self.device}")

		self.load_data()  		
		self.build_network()
		self.build_optimizer()


	def load_data(self):
		# load mean, std, xmean, xstd
		data = np.load(self.directory + "mean_and_std.npz")
		self.mean = data['mean']
		self.std = data['std']

		# load the whole motion sequence from rdc
		self.rdc.set_mocap_input()
		self.head = deepcopy(self.rdc.head) 
		self.imu_rot_input = deepcopy(self.rdc.imu_rot_input)
		self.imu_acc_input = deepcopy(self.rdc.imu_acc_input)

		# contact manager
		self.contact = {}
		self.contact[0] = 0.0
		self.contact_manager = ContactManagerCustomRun(contact_dict=self.contact)
		self.rdc.camera_head_signal.load_pcd_result()
		pcd_points = np.asarray(self.rdc.camera_head_signal.pcd.points)
		self.contact_manager.add_pcd_points(pcd_points=pcd_points)

		# skel 
		body_model = amass.load_body_model(motion_constants.BM_PATH)
		fairmotion_skel, _ = amass.create_skeleton_from_amass_bodymodel(bm=body_model)
		self.skel = fairmotion_skel
		self.skel_offset = fairmotion_skel.get_joint_offset_list()
		self.skel_parent = fairmotion_skel.get_parent_index_list()

		self.skel_offset = torch.from_numpy(self.skel_offset[np.newaxis, np.newaxis, ...]).to(self.device).float() 		# expand skel offset into tensor

	def fetch_current_sequence(self, start_frame):
		window = self.config['custom']['seq_len']
		frame_key, contact_height = get_height_offset_current_frame(self.contact, start_frame)

		head_T_window = self.head[start_frame:start_frame+window]
		imu_rot_window = self.imu_rot_input[start_frame:start_frame+window]
		imu_acc_window = self.imu_acc_input[start_frame:start_frame+window]

		head_T = head_T_window[np.newaxis, ...]
		imu_rot = imu_rot_window[np.newaxis, ...]
		imu_acc = imu_acc_window[np.newaxis, ...]

		# get head up vector before normalization
		height_indice = 1 if motion_constants.UP_AXIS == "y" else 2
		upvec_axis = np.array([0,0,0]).astype(dtype=np.float32)
		upvec_axis[1] = 1.0

		# get head up vector before normalization (here apply z!)
		head_upvec = np.einsum('ijkl,l->ijk', head_T[...,:3,:3], upvec_axis) # fixed bug! 
		head_height = head_T[...,height_indice,3][..., np.newaxis] - contact_height

		# normalize by head 
		batch, seq_len, _, _ = head_T.shape
		head_start_T = head_T[:,0:1,...] # [# window, 1, 4, 4]
		head_invert = invert_T(head_start_T)
		normalized_head_T = head_invert @ head_T

		# normalize imu (rot acc) by head 
		head_invert_rot = head_invert[...,:3,:3] 
		head_invert_rot = head_invert_rot[:,:,np.newaxis, ...]
		normalized_imu_rot = head_invert_rot @ imu_rot  # [Window #, seq, 2, 3, 3]
		normalized_imu_acc = np.einsum('ijklm,ijkm->ijkl', head_invert_rot, imu_acc) # [Window #, seq, 2, 3]
		normalized_imu_concat = T_to_6d_and_pos(conversions.Rp2T(normalized_imu_rot, normalized_imu_acc)) # [Window #, seq, 2, 9]
		normalized_imu_concat = normalized_imu_concat.reshape(batch, seq_len, -1)

		# concat into network input form
		normalized_head = T_to_6d_and_pos(normalized_head_T)
		head_imu_input = np.concatenate((head_height, head_upvec, normalized_head, normalized_imu_concat), axis=-1) 

		self.input_seq_dim = head_imu_input.shape[-1]

		# normalize input
		head_imu_input_normalized = (head_imu_input - self.mean) / self.std 
		# embed()

		input_ = {}
		input_['input_seq'] = torch.Tensor(head_imu_input_normalized).float()
		input_['head_start'] = torch.Tensor(head_start_T[:,:,np.newaxis,...]).float()
		input_['seq_len'] = window
		input_['normalized_head_pos'] = torch.Tensor(normalized_head_T[...,:3,3])

		return input_


	def build_network(self):
		logging.info(f"Loading model...")
		self.model_dir = os.path.join(self.directory, "model/")

		data_dict = {}
		data_dict['input_dim'] = 31
		data_dict['mid_dim'] = 12
		data_dict['output_dim'] = 135

		model_dict = self.config['model']

		self.model = imu2body.model.load_model(data_config=data_dict, model_config=model_dict)
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


	def run(self):
		self.model.eval()

		self.motion_data = []
		select_idx = 0

		frame_start_idx = 0
		while True:
			if frame_start_idx >= self.head.shape[0]:
				break # end of whole sequence

			sampled_batch = self.fetch_current_sequence(frame_start_idx)
			with torch.no_grad():
				# print(f"start with start frame: {frame_start_idx}")

				input_seq = sampled_batch['input_seq'].to(self.device)
				output_tuple = self.model(input_seq) # hand (mid), foot, final_output (body)

				# get results
				gt_tuple = None
				results = self.get_loss(output_tuple=output_tuple, gt_tuple=gt_tuple)
				output_root, output_joint_rot, output_contact_labels = results

				# convert to renderdata
				rd = MotionData(gt_root=None, \
				gt_rot=None, \
				output_root=output_root[select_idx].detach().cpu().numpy(),\
				output_rot=output_joint_rot[select_idx].detach().cpu().numpy(),
				use_gt=False
				)

				start_T = sampled_batch['head_start']
				rd.convert_to_matrix(start_T=start_T[select_idx], use_gt=False)
				binary_contact_labels = output_contact_labels[select_idx].detach().cpu().numpy() > 0.5
				rd.set_contact_label(binary_contact_labels)

				contact_frame = self.contact_manager.update(rd=rd, frame_start=frame_start_idx)

				if contact_frame is None:
					frame_start_idx = frame_start_idx + self.config['custom']['offset']
					rd.set_frame(parse_frame=self.config['custom']['offset'])
				else:
					seq_end = contact_frame - frame_start_idx
					rd.set_frame(parse_frame=seq_end)
					frame_start_idx = contact_frame

				# remove unnecessary parts in the contact history (overlapping parts)
				contact_history_frames_to_remove = rd.seq_len - rd.parse_frame
				self.contact_manager.parse_contact_history(contact_history_frames_to_remove) 

				self.motion_data.append(rd)

		write_motion_data_result(motion_data_list=self.motion_data, \
		   				 save_dir=self.directory, \
						 filename=self.custom_filename, \
						 custom_config=self.config['custom'], \
						 contact_manager=self.contact_manager)


	def get_loss(self, output_tuple, gt_tuple):
		mid_output, contact_labels, output_seq = output_tuple # [1, seq_len, 12] [1, seq_len, 2] [1, seq_len, 135]
		batch, seq_len, _ = output_seq.shape

		if gt_tuple is not None:
			mid_seq = gt_tuple['mid_seq'].to(self.device)
			tgt_seq = gt_tuple['tgt_seq'].to(self.device) # [batch, seq_len, dim]
			global_pos = gt_tuple['global_p'].to(self.device)

		output_root = output_seq[...,:3]
		output_joint_rot = output_seq[...,3:]
		output_joint_rot = output_joint_rot.reshape(batch, seq_len, -1, 6)

		contact_labels = torch.sigmoid(contact_labels) # only for inference 
		return (output_root, transforms.rotation_6d_to_matrix(output_joint_rot), contact_labels) 

	def get_loss_v2(self, output_tuple, gt_tuple):
		mid_output, contact_labels, output_seq = output_tuple # [1, seq_len, 12] [1, seq_len, 2] [1, seq_len, 135]
		batch, seq_len, _ = output_seq.shape

		output_joint_rot = output_seq.reshape(batch, seq_len, -1, 6)
		contact_labels = torch.sigmoid(contact_labels)

		normalized_head_pos = gt_tuple['normalized_head_pos']

		# set root to zero
		output_root = torch.zeros(size=(batch, seq_len,3)).to(self.device)

		# perform forward kinematics
		output_pos_mat = rot_matrix_fk_tensor(transforms.rotation_6d_to_matrix(output_joint_rot), output_root, self.skel_offset, self.skel_parent)

		# get mapping that sends to head (per frame)
		cur_head_pos = output_pos_mat[...,motion_constants.head_joint_idx, :]
		head_mapping = normalized_head_pos.to(self.device) - cur_head_pos

		# map root from zero to global -> set that to output_root 
		output_root = output_root + head_mapping 
		
		return (output_root, transforms.rotation_6d_to_matrix(output_joint_rot), contact_labels)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--test_name", type=str, default="")
	parser.add_argument("--rdc-folder", type=str, default="")
	parser.add_argument("--bm-type", type=str, default="smplx")

	# deprecated
	parser.add_argument("--custom_datapath", type=str, default="")
	parser.add_argument("--offset-size", type=int, help="# of frames to slide per window", default=0)


	args = parser.parse_args()
	
	custom_run = IMU2BodyRealdataRun(args=args)
	custom_run.run()

