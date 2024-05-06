from copy import deepcopy
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
# from bvh import *
from autoencoder_v2.autoencoder_utils import *
from fairmotion.core import motion as motion_classes
from fairmotion.ops import conversions
import imu2body.amass as amass 
from fairmotion.utils import utils
import constants.motion_data as motion_constants
from interaction.contact import *
from fairmotion.ops.math import *

logging.basicConfig(
	format="[%(asctime)s] %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	level=logging.INFO,
)

bm_path = "../data/smpl_models/smplx/SMPLX_NEUTRAL.npz"

def get_2d_pos_rot_vel(root_T, v_face_skel, v_up_env):
	# rotations = motion.rotations() # [F, 22, 3, 3]
	# global_positions = motion.positions(local=False) # [F, 22, 3]
	# root_pos = deepcopy(global_positions[:,0:1,:]) # [F, 1, 3]

	# root_plane = plane*root_pos  # [F, 1, 3]
	# root_plane = root_plane.reshape(-1, 3)   # [F, 3]
	# velocity = root_plane[1:] - root_plane[:-1]  # [F-1, 3]

	# height = root_pos[:,0,height_indice][..., np.newaxis]   # [F, 1]

	# _ , global_p = quat_fk(motion.quats, motion.rootPos, motion.skel.offset, motion.skel.parent)
	# c_l, c_r = extract_feet_contacts(global_p, [3,4], [7,8], velfactor=0.05) # [F, 2] for each
	# c_lr = np.concatenate((c_l, c_r), axis=-1)  #  [124, 4]

	# root_pos = deepcopy(motion.rootPos) # [F, 1, 3]
	# joint_Q = deepcopy(motion.quats) # [F, 22, 4]

	# get root position differences in xz coordinates
	# root_xz = np.array([1,0,1])*root_pos  # [F, 1, 3]
	# root_xz = root_xz.reshape(-1, 3)   # [F, 3]
	# velocity = root_xz[1:] - root_xz[:-1]  # [F-1, 3]

	# root height (y)
	# height = root_pos[:,0,height_indice][..., np.newaxis]   # [F, 1]

	# for all frames make the root face the -y
	facing_dir_env = np.array([0,0,1]) if motion_constants.UP_AXIS == "y" else np.array([0,-1,0])
	plane = np.array([1,0,1]) if motion_constants.UP_AXIS == "y" else np.array([1,1,0])
	idx = 1 if motion_constants.UP_AXIS == "y" else 2

	root_rot, root_pos = conversions.T2Rp(root_T)

	root_plane = plane*root_pos  # [F, 3]
	velocity = root_plane[1:] - root_plane[:-1]  # [F-1, 3]

	forward = plane[np.newaxis, :] * np.einsum('ijk,k->ij', root_rot, v_face_skel)
	facing_rotations = R_from_vectors_tensor(forward, facing_dir_env[np.newaxis,...])
	normalized_root_rot = facing_rotations @ root_rot # send root to facing_dir_env

	velocity = facing_rotations[1:] @ velocity[..., np.newaxis]
	facing_rot_vel_rotmat = facing_rotations[1:] @ facing_rotations[:-1].swapaxes(-2,-1) # TODO check transpose correct
	facing_rot_vel_aa = conversions.R2A(facing_rot_vel_rotmat)[...,idx]

	# for all frames make the root Q face the z axis
	# root_Q = deepcopy(joint_Q[:,0,:])   # [F, 4]
	# forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, :] \
	# 			* quat_mul_vec(root_Q, np.array([0, 0, 1])[np.newaxis, np.newaxis, :]) # skeleton facing dir is the z-axis
	# forward = normalize(forward).reshape(-1, 3) # dir is 3 dimensional vector (F, 3)
	# facing_rotations = quat_normalize(quat_between(forward, np.array([0,0,1]))) # send forward to z (F, 4)
	# joint_Q[:,0,:] = quat_mul(facing_rotations, root_Q) # send to z

	# velocity = quat_mul_vec(facing_rotations[1:], velocity)
	# facing_rot_diff = quat_to_angle_axis(quat_mul(facing_rotations[1:], quat_inv(facing_rotations[:-1])))[..., 1] 
	
	# copy the first frame vel of the velocity array (assume that vel is same at t=0 and t=1)
	velocity = np.concatenate((velocity[0:1], velocity), axis=0) # assume that first axis is frame / shape: [F, 3]
	facing_rot_vel_aa = np.concatenate((facing_rot_vel_aa[0:1], facing_rot_vel_aa), axis=0) # assume that first axis is frame
	facing_rot_vel_aa = facing_rot_vel_aa[..., np.newaxis]   # [F, 1]

	return velocity, facing_rot_vel_aa, normalized_root_rot

def load_data_from_amass_autoencoder(base_dir, file_list, debug=False):

	assert isinstance(file_list, list), "Always a list of filenames should be given. If custom, should be given as [filename] format."	
	assert len(file_list) > 0, "There should be more than one file in the file list"

	filepath_list = [os.path.join(base_dir, file) for file in file_list]
	num_cpus = min(24, len(file_list)) if not debug else 1

	npz_files = [f for f in filepath_list if f.endswith('.npz')]
	bvh_files = [f for f in filepath_list if f.endswith('.bvh')]

	# this only works when the list is npz (amass data)
	# read skel
	body_model = amass.load_body_model(bm_path=bm_path)
	skel_with_offset = amass.create_skeleton_from_amass_bodymodel(bm=body_model)	
	skel = skel_with_offset[0]

	v_face_skel = skel.v_face
	v_up_env = skel.v_up_env 

	assert utils.axis_to_str(v_up_env) == motion_constants.UP_AXIS, "v_up_env of skel should be same as up axis in motion constants!"

	facing_dir_env = np.array([0,0,1]) if motion_constants.UP_AXIS == "y" else np.array([0,-1,0])
	plane = np.array([1,0,1]) if motion_constants.UP_AXIS == "y" else np.array([1,1,0])
	idx = 1 if motion_constants.UP_AXIS == "y" else 2

	# motion_list = utils.run_parallel(amass.create_motion_from_amass_data, filepath_list, num_cpus=num_cpus, bm=body_model, skel_with_offset=deepcopy(skel_with_offset))
	motion_list = utils.run_parallel(amass.create_motion_from_amass_data, npz_files, num_cpus=num_cpus, bm=body_model, skel_with_offset=deepcopy(skel_with_offset))

	# constants
	window = motion_constants.preprocess_window * 2
	offset = motion_constants.preprocess_offset
	height_indice = 1 if motion_constants.UP_AXIS == "y" else 2
	plane = np.array([1,0,1]) if motion_constants.UP_AXIS == "y" else np.array([1,1,0])

	if len(bvh_files) > 0:
		motion_list_bvh = utils.run_parallel(amass.bvh_to_amass_motion, bvh_files, num_cpus=num_cpus, amass_skel=deepcopy(skel))
		motion_list += motion_list_bvh

	logging.info(f"Done converting amass into fairmotion Motion class")


	# read list
	local_T = []
	global_T = []
	facing_dir_vel = []
	facing_pos_vel = []
	c_lr = []

	total_num = len(motion_list)
	for idx, motion in enumerate(motion_list):
		if motion is None or motion.num_frames() < window:
			continue
		motion_local_T = motion.to_matrix()
		motion_global_T = motion.to_matrix(local=False)

		# facing_pos_vel_motion, facing_dir_vel_motion, normalized_root_rot = get_2d_pos_rot_vel(motion_global_T[:,0], v_face_skel=v_face_skel, v_up_env=v_up_env)
		# motion_local_T[:,0,:3,:3] = normalized_root_rot

		# split into sliding windows
		i = 0
		while True:
			if i+window > motion_local_T.shape[0]:
				break
			else:
				local_T_window = motion_local_T[i:i+window]
				global_T_window = motion_global_T[i: i+window]

			contact_labels = generate_contact_labels(global_T=global_T_window, only_foot=True)

			# record
			local_T.append(local_T_window)
			global_T.append(global_T_window)
			c_lr.append(contact_labels)

			i += offset

	# done reading 
	local_T = np.asarray(local_T).astype(dtype=np.float32) # [# seq, seq_len, J, 4, 4]
	global_T = np.asarray(global_T).astype(dtype=np.float32) # [# seq, seq_len, J, 4, 4]
	c_lr = np.asarray(c_lr).astype(dtype=np.float32)
	c_lr = c_lr.transpose(0,2,1)

	batch, seq_len, num_joints, _, _ = local_T.shape
	# facing_dir_vel = np.asarray(facing_dir_vel).astype(dtype=np.float32) # [# seq, seq_len, 1]
	# facing_pos_vel = np.asarray(facing_pos_vel).astype(dtype=np.float32) # [# seq, seq_len, J, 3, 1]

	forward = plane[np.newaxis,...] * (np.einsum('ijk,k->ij',global_T[...,0,0,:3,:3],v_face_skel))
	facing_rotations = R_from_vectors_tensor(forward, facing_dir_env) # send forward to facing_dir_env

	normalized_root_start_rot = facing_rotations[:,np.newaxis, np.newaxis, ...] @ global_T[:,0:1,0:1,:3,:3]
	normalized_root_start_pos = np.zeros_like(global_T[...,0:1,0:1,:3,3])
	normalized_root_start_pos[...,height_indice] = global_T[:,0:1,0:1,height_indice,3] # only height left
	normalized_root_start_T = conversions.Rp2T(normalized_root_start_rot, normalized_root_start_pos)

	cur_to_normalized = normalized_root_start_T @ invert_T(global_T[:,0:1,0:1,...])

	# embed()
	# normalized_local_T = deepcopy(local_T)
	normalized_global_T = deepcopy(global_T)
	
	# loop to save ram space
	for i in range(seq_len):
		# n_l_t = cur_to_normalized @ normalized_local_T[:,i:i+1,0:1,:,:]
		# normalized_local_T[:,i:i+1,0:1,:,:] = n_l_t
		g_t = cur_to_normalized @ normalized_global_T[:,i:i+1,...]
		normalized_global_T[:,i:i+1,...] = g_t

	del global_T

	local_T[...,0:1,:,:] = cur_to_normalized @ local_T[...,0:1,:,:]
	# normalized_global_T = cur_to_normalized @ normalized_global_T

	normalized_to_cur = invert_T(cur_to_normalized)

	if debug:
		return local_T, normalized_global_T, c_lr, normalized_to_cur, normalized_root_start_T
	
	# input: contact, root pos, joint rotation in 6d representation
	joint_rot = T_to_6d_rot(local_T) # [#, seq_len, 22, 6]
	joint_rot = joint_rot.reshape(batch, seq_len, -1)
	root_pos = local_T[...,0,:3,3] # [#, seq_len, 3]

	input_seq = np.concatenate((c_lr, root_pos, joint_rot), axis=-1)

	# for losses
	global_p = normalized_global_T[...,:3,3]


	return input_seq, global_p, local_T[...,:3,:3], c_lr, normalized_to_cur



def load_data_with_args(fnames, bvh_list, args):
	logging.info(f"Start processing {fnames} data with {len(bvh_list)} files...")
	data , total_len = load_data(bvh_list, base_dir=args.base_dir)
	logging.info(f"Processed {fnames} data with {total_len} sequences")
	pickle.dump(data, open(os.path.join(args.preprocess_path, f"{fnames}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
	logging.info(f"Saved {fnames} data with {total_len} sequences")
	del data 

def load_data(file_list, base_dir=""):

	# head_imu_input, ee_pos, output, global_p, local_rot, head_start, c_lr = load_data_from_amass_autoencoder(base_dir, file_list, custom_config=custom_config)
	input_seq, global_p, local_rot, c_lr, normalized_to_cur = load_data_from_amass_autoencoder(base_dir, file_list)

	total, seq_len, _  = input_seq.shape

	input_ = {}
	input_['input_seq'] = input_seq
	# input_['tgt_seq'] = input_seq # autoencoder: tgt is same as input
	input_['global_p'] = global_p 
	input_['contact_label'] = c_lr
	input_['root'] = global_p[...,0,:]
	input_['local_rot'] = local_rot
	input_['root_start'] = normalized_to_cur

	# Get test-set for windows of 120 frames, offset by 10 frames
	# signal, output, global_p, local_rot, head_start = load_data_from_amass(base_dir, bvh_list, \
	# 																window=seq_len, offset=offset)		

	# set necessary information to dictionary
	# input_ = {}
	# input_['input_seq'] = head_imu_input 
	# input_['mid_seq'] = ee_pos 
	# input_['tgt_seq'] = output 
	# input_['global_p'] = global_p
	# input_['root'] = global_p[..., 0, :] 
	# input_['local_rot'] = local_rot
	# input_['head_start'] = head_start 
	# input_['contact_label'] = c_lr

	return input_, total

def parse_filenames_and_load(args):

	def split_list(input_list, parts):
		length = len(input_list)
		return [ input_list[i*length // parts: (i+1)*length // parts] 
				for i in range(parts) ]

	fnames_list = ['train', 'test', 'validation']
	train_split_num = 12
	test_split_num = 2

	if not os.path.exists(args.preprocess_path):
		utils.create_dir_if_absent(args.preprocess_path)

	fnames = args.data_type
	if args.data_type in fnames_list:
		file = open(os.path.join(args.data_config_path, f'{fnames}_fnames.txt'), 'r')
		filename_list = file.read().split("\n")

	if fnames == "train":
		train_list_split = split_list(filename_list, train_split_num)			
		train_idx = args.file_idx
		# for i in range(train_split_num):
		fnames_edit = f"{fnames}_{train_idx+1}"
		load_data_with_args(fnames=fnames_edit, bvh_list=train_list_split[train_idx], args=args)

	elif fnames == "test":
		test_list_split = split_list(filename_list, test_split_num)			
		test_idx = args.file_idx
		fnames_edit = f"{fnames}_{test_idx+1}"
		load_data_with_args(fnames=fnames_edit, bvh_list=test_list_split[test_idx], args=args)

	elif fnames == "validation":
		load_data_with_args(fnames=fnames, bvh_list=filename_list, args=args)

	else:
		# save config text files
		print("save config text files")
		dest = os.path.join(args.preprocess_path, "data_config")
		utils.create_dir_if_absent(dest)
		config_copy_command = f"cp -r {args.data_config_path} {dest}"
		os.system(config_copy_command)

	# for fnames in fnames_list:
	# 	file = open(os.path.join(args.data_config_path, f'{fnames}_fnames.txt'), 'r')
	# 	filename_list = file.read().split("\n")

	# 	if fnames == "train":
	# 		train_list_split = split_list(filename_list, train_split_num)			
	# 		train_idx = args.file_idx
	# 		# for i in range(train_split_num):
	# 		fnames_edit = f"{fnames}_{train_idx+1}"
	# 		load_data_with_args(fnames=fnames_edit, bvh_list=train_list_split[train_idx], args=args)
	# 		sys.exit(0)
	# 	elif fnames == "test":
	# 		load_data_with_args(fnames=fnames, bvh_list=filename_list, args=args)
		



if __name__ == "__main__":

	# add argparse
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--base-dir",
		type=str,
		required=True,
	)
	parser.add_argument(
		"--data-config-path",
		type=str,
		required=True
	)
	parser.add_argument(
		"--preprocess-path",
		type=str,
		default=True
	)

	parser.add_argument(
		"--data-type",
		type=str,
		required=True
	)
	parser.add_argument(
		"--file-idx",
		type=int,
		default=-1
	)
	args = parser.parse_args()

	# if args.preprocess_path is None:
	# 	args.preprocess_path = os.path.join(args.base_dir, "amass")
	
	# for generating preprocessed pkl files
	parse_filenames_and_load(args)

	# for debugging
	# result = load_data_from_amass_autoencoder("../data/amass", ["CMU/01/01_01_stageii.npz"], debug=False)
