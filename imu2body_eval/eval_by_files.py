from copy import deepcopy
from matplotlib.pyplot import axes
import torch
import sys, os
# sys.path.insert(0, os.path.dirname(__file__))
# sys.path.append("..")
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import logging
import numpy as np
import os
import pickle

from IPython import embed
from pytorch3d import transforms
# from bvh import *
from fairmotion.core import motion as motion_classes
from fairmotion.ops import conversions, math as fairmotion_math
from fairmotion.data import bvh
import sys
from datetime import datetime
from copy import deepcopy
from imu2body_eval.functions import *
# from visualizer import bvh_single_visualizer
import imu2body_eval.amass_smplh as amass_smplh
from fairmotion.utils import utils
from tqdm import tqdm
from copy import deepcopy
import constants.imu as imu_constants
import constants.motion_data as motion_constants
import imu2body_eval.imu as imu
# from interaction.contact import *
from IPython import embed 
from tqdm import tqdm
# for totalcapture data
from datasets.tc_data import * 
from datasets.hps_data import * 

logging.basicConfig(
	format="[%(asctime)s] %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	level=logging.INFO,
)

bm_path = "../data/smpl_models/smplh/male/model.npz"


def load_data_from_amass(base_dir, file_list, save_path, debug=False):
	assert isinstance(file_list, list), "Always a list of filenames should be given. If custom, should be given as [filename] format."	
	assert len(file_list) > 0, "There should be more than one file in the file list"

	filepath_list = [os.path.join(base_dir, file) for file in file_list]
	num_cpus = min(24, len(file_list)) if not debug else 1

	npz_files = [f for f in filepath_list if f.endswith('.npz')]
	bvh_files = [f for f in filepath_list if f.endswith('.bvh')]
	pkl_files = [f for f in filepath_list if f.endswith('.pkl')] # this is for hps data

	# this only works when the list is npz (amass data)
	# read skel
	body_model = amass_smplh.load_body_model(bm_path=motion_constants.SMPLH_BM_PATH)
	skel_with_offset = amass_smplh.create_skeleton_from_amass_bodymodel(bm=body_model)	
	skel = skel_with_offset[0]

	motion_list = utils.run_parallel(amass_smplh.create_motion_from_amass_data, npz_files, num_cpus=num_cpus, bm=body_model, skel_with_offset=deepcopy(skel_with_offset))
	
	if len(bvh_files) > 0:
		motion_list_bvh = utils.run_parallel(amass_smplh.bvh_to_amass_motion, bvh_files, num_cpus=num_cpus, amass_skel=deepcopy(skel))
		motion_list += motion_list_bvh
		npz_files += bvh_files
	
	if len(pkl_files) > 0:
		motion_list = utils.run_parallel(load_motion_and_scene, pkl_files, num_cpus=num_cpus, only_read_motion=True)
		npz_files += pkl_files

	dataset_folder_list = [s.split('/')[3] for s in npz_files]
	logging.info(f"Done converting amass into fairmotion Motion class")
	
	ee_joint_names = imu_constants.imu_joint_names + motion_constants.FOOT_JOINTS
	ee_joint_idx = [skel.get_index_joint(jn) for jn in ee_joint_names]

	imu_joint_names = imu_constants.imu_joint_names
	imu_joint_idx = [skel.get_index_joint(jn) for jn in imu_joint_names]

	# constants
	window = motion_constants.preprocess_window
	offset = motion_constants.preprocess_window
	height_indice = 1 if motion_constants.UP_AXIS == "y" else 2

	is_custom_run = True

	for idx, motion in enumerate(tqdm(motion_list)):
		# read list
		local_T = [] 
		global_T = []

		# imu signal list
		imu_rot = []
		imu_acc = []

		# contact labels
		c_lr = []

		if motion is None or motion.num_frames() < window:
			continue
		motion_local_T = motion.to_matrix()
		motion_global_T = motion.to_matrix(local=False)

		# for totalcapture: replace to real imu signals 
		if 'TotalCapture' in npz_files[idx]:
			orig_pose_filename = npz_files[idx].split("/")
			orig_pose_filename = '_'.join(orig_pose_filename[-2:])
			tc_filename = orig_pose_filename.replace("_poses.npz", ".pkl")
			motion_imu_rot, motion_imu_acc = get_imu_from_tc(tc_filename)
			
			# there is 1-2 frame difference in TC amass and TC DIP
			diff = abs(motion.num_frames() - motion_imu_rot.shape[0])
			# print(f"diff: {diff} name: {orig_pose_filename}")
			if diff > 10: #  s5_freestyle3 
				continue
		else:
			motion_imu_rot, motion_imu_acc = imu.imu_from_global_T(motion_global_T, imu_joint_idx)

		# set contact/height offset 
		height_offset = 0.0
		contact_frame = 0
		contact = {}
		contact[contact_frame] = height_offset 

		# split into sliding windows
		start_frame_label = []

		i = 0
		while True:
			if i >= motion_local_T.shape[0]:
				break 
			if i+window >= motion_local_T.shape[0]:
				i = motion_local_T.shape[0] - window
			else:
				local_T_window = motion_local_T[i: i+window]
				global_T_window = motion_global_T[i: i+window]
				imu_rot_window = motion_imu_rot[i: i+window]
				imu_acc_window = motion_imu_acc[i: i+window]

			# no height adjust in eval
			local_T_window_height_adjust = deepcopy(local_T_window)
			global_T_window_height_adjust = deepcopy(global_T_window)

			# record
			local_T.append(local_T_window_height_adjust)
			global_T.append(global_T_window_height_adjust)
			imu_rot.append(imu_rot_window)
			imu_acc.append(imu_acc_window)

			# record start frame idx
			start_frame = i
			start_frame_label.append(start_frame)

			i += offset

		
		# do it per motion
		local_T = np.asarray(local_T).astype(dtype=np.float32) # [# of window, window size, J, 4, 4]
		global_T = np.asarray(global_T).astype(dtype=np.float32)
		imu_rot = np.asarray(imu_rot).astype(dtype=np.float32) 
		imu_acc = np.asarray(imu_acc).astype(dtype=np.float32)

		# c_lr = np.asarray(c_lr).astype(dtype=np.float32)
		# c_lr = c_lr.transpose(0,2,1)

		# preprocess and add sensor offset
		head_idx = skel.get_index_joint("Head")

		upvec_axis = np.array([0,0,0]).astype(dtype=np.float32)
		upvec_axis[1] = 1.0 # upvec is y even in amass

		head_upvec = np.einsum('ijkl,l->ijk', global_T[..., head_idx,:3,:3], upvec_axis) # fixed bug! 
		head_height = global_T[...,head_idx,height_indice,3][..., np.newaxis]

		# by head 
		head_start_T = global_T[:,0:1,head_idx:head_idx+1,...] # [# window, 1, 1, 4, 4]
		batch, seq_len, num_joints, _, _ = local_T.shape
		head_invert = invert_T(head_start_T)
		local_T[...,0:1,:,:] = head_invert @ local_T[...,0:1,:,:] # only adjust root

		# loop to save ram space..
		normalized_global_T = np.zeros(shape=global_T.shape)
		for i in range(seq_len):
			g_t = head_invert @ global_T[:,i:i+1,...]
			normalized_global_T[:,i:i+1,...] = g_t

		del global_T

		# imu & head input
		head_invert_rot = head_invert[...,:3,:3] 
		normalized_imu_rot = head_invert_rot @ imu_rot  # [Window #, seq, 2, 3, 3]
		normalized_imu_acc = np.einsum('ijklm,ijkm->ijkl', head_invert_rot, imu_acc) # [Window #, seq, 2, 3]
		normalized_imu_concat = T_to_6d_and_pos(conversions.Rp2T(normalized_imu_rot, normalized_imu_acc)) # [Window #, seq, 2, 9]
		normalized_imu_concat = normalized_imu_concat.reshape(batch, seq_len, -1)

		normalized_head = T_to_6d_and_pos(normalized_global_T[...,head_idx, :, :])
		head_imu_input = np.concatenate((head_height, head_upvec, normalized_head, normalized_imu_concat), axis=-1) 

		# mid (output of 1st network, input of 2nd network)
		ee_pos = normalized_global_T[...,ee_joint_idx, :3, 3]	
		reshaped_ee_pos = np.transpose(ee_pos, (1, 2, 0, 3))
		ee_pos_v = reshaped_ee_pos.reshape(batch, seq_len, -1)

		if debug:
			return normalized_imu_rot, normalized_imu_acc, ee_pos_v, local_T, normalized_global_T, head_start_T # invert_T(head_invert)

		# change into output sequence form by concatenating root pos 3d and (root-included) joint rotations (6d)
		local_rotation_6d = T_to_6d_rot(local_T)
		local_rotation_6d = local_rotation_6d.reshape(batch, seq_len, -1)

		output = np.concatenate((normalized_global_T[...,0,:3,3], local_rotation_6d), axis=-1) # [# of windows, seq_len, 6J+3]	

		# return global pos for FK loss calc
		global_p = normalized_global_T[...,:3,3]

		total, seq_len, _  = output.shape
		result_dict = {}
		result_dict['input_seq'] = torch.Tensor(head_imu_input).float() 
		result_dict['mid_seq'] = torch.Tensor(ee_pos_v).float()
		result_dict['tgt_seq'] = torch.Tensor(output).float() 
		result_dict['global_p'] = torch.Tensor(global_p).float()
		result_dict['root'] = torch.Tensor(global_p[..., 0, :]).float() 
		result_dict['local_rot'] = torch.Tensor(local_T[...,:3,:3]).float()
		result_dict['head_start'] = torch.Tensor(head_start_T) 
		result_dict['contact_label'] = torch.Tensor(c_lr).float()
		result_dict['start_frame'] = start_frame_label
		result_dict['total_length'] = motion_local_T.shape[0]

		filename = npz_files[idx]
		result_dict['filename'] = filename 

		# save
		dataset_folder = dataset_folder_list[idx]
		save_path_per_file = os.path.join(save_path, dataset_folder)
		utils.create_dir_if_absent(save_path_per_file)

		with open(os.path.join(save_path_per_file, f"{idx}.pkl"), "wb") as file:
			pickle.dump(result_dict, file, protocol=pickle.HIGHEST_PROTOCOL)



def load_filelist(args):
	test_txt_filename = ""
	if args.data_type == "amass_vr":
		test_txt_filename = "amass_vr_fnames.txt"
	if args.data_type == "tc":
		test_txt_filename = "tc_fnames.txt"
	if args.data_type == "hps":
		test_txt_filename = "hps_fnames.txt"

	file = open(os.path.join(args.data_config_path, test_txt_filename), 'r')
	# copy this to args.save_path 
	utils.create_dir_if_absent(args.save_path)
	os.system(f'cp {os.path.join(args.data_config_path, test_txt_filename)} {os.path.join(args.save_path, "test_fnames.txt")}')
	filename_list = file.read().split("\n")
	load_data_from_amass(base_dir=args.base_dir, file_list=filename_list, save_path=args.save_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--base-dir",
		type=str,
		required=True
	)

	parser.add_argument(
		"--data-config-path",
		type=str,
		required=True
	)

	parser.add_argument(
		"--save-path",
		type=str,
		required=True
	)
	parser.add_argument(
		"--data-type",
		type=str,
		required=True
	)

	args = parser.parse_args()
	load_filelist(args=args)