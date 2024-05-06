import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import dateutil.parser
from scipy import ndimage 
from copy import deepcopy
from fairmotion.ops import conversions, motion as motion_ops
from fairmotion.data import bvh
from realdata.watch_sensor_parser import *
import xml.dom.minidom
from imu2body.imu import * 
from imu2body.functions import *
from realdata.camera_sensor_parser import *
import yaml 
import argparse
# from realdata.mocap_data_parser import *
from realdata.utils import *

from datetime import datetime, timedelta, timezone
import constants.path as path_constants

class RealDataCollection2(object):
	def __init__(self, folder_path, no_calib=False) -> None:

		self.config_folder_path = os.path.join(path_constants.DATA_PATH, folder_path)
		self.data_folder_path = os.path.join(path_constants.DATA_PATH, folder_path)
		
		self.include_clap = False
		self.manual_sync = False
	
		# read config
		self.parse_config()

		self.multi_person = False
		if 'multi' in self.config:
			self.multi_person = True
		# read watch
		self.watch_signal = {}
		self.wrist_joints = ["LeftHand", "RightHand"]
		for i, joint in enumerate(self.wrist_joints):
			sensorlog_filepath = os.path.join(self.data_folder_path, f"watch_sensorlog_{i+1}.csv")
			if os.path.exists(sensorlog_filepath):
				time_offset = None
				# self.watch_signal[joint] = WatchIMUSignal2(os.path.join(self.data_folder_path, f"watch_sensorlog_{i+1}.csv"), lr=i, time_offset= self.config['watch']['offset'][i])
				if 'watch' in self.config:
					time_offset = self.config['watch']['offset'][i]
					self.manual_sync = True
				self.watch_signal[joint] = WatchIMUSignal2(os.path.join(self.data_folder_path, f"watch_sensorlog_{i+1}.csv"), lr=i, time_offset=time_offset)
			else:
				self.watch_signal[joint] = None
		
		# load camera
		self.camera_head_signal = CameraHeadSignal2(dir=folder_path)

		# set timecode
		if sys.version_info < (3, 8):
			import pickle5 as pickle 
		else:
			import pickle
		self.calib_timecode = {}
		self.calib_timecode['watch'] = pickle.load(open(os.path.join(self.data_folder_path, "calib.pkl"), "rb"))
		self.calib_timecode['camera'] = pickle.load(open(os.path.join(self.data_folder_path, "camera_calib.pkl"), "rb"))
		
		# camera timecode
		if self.multi_person:
			idx = self.config['multi']['idx']
			if self.calib_timecode['camera'][idx]['shutter'] is None:
				self.calib_timecode['camera'][idx]['shutter'] = self.config['cam_shutter_timestamp']
		else:
			if self.calib_timecode['camera']['shutter'] is None:
				self.calib_timecode['camera']['shutter'] = self.config['cam_shutter_timestamp']

		if self.multi_person:
			calib_dict = self.calib_timecode['camera'][self.config['multi']['idx']]
			calib_dict['clap'] = self.calib_timecode['camera']['clap']
		else:
			calib_dict = self.calib_timecode['camera']

		if no_calib:
			self.calib_dict = calib_dict
			return 

		# calibration
		self.camera_head_signal.set_time_config(calib_dict=calib_dict)
		self.calib_watch()
		self.calib_camera()

	def parse_config(self):
		config_path = f"{self.config_folder_path}/config.yaml"
		self.config = yaml.safe_load(open(config_path, 'r').read())


	def calib_watch(self):
		for i, joint in enumerate(self.wrist_joints):
			calib_key = f"origin_{i+1}"
			if calib_key not in self.calib_timecode['watch']:
				calib_key = "origin"
			if self.watch_signal[joint] is None:
				continue
			self.watch_signal[joint].calib_origin(self.calib_timecode['watch'][calib_key]) # origin calib
			self.watch_signal[joint].calib_tpose(tpose_timecode_range=self.calib_timecode['watch']['tpose'], tpose_joint_ori=motion_constants.joint_ori_tpose) # tpose calib

	def calib_camera(self):
		if self.camera_head_signal.empty:
			return 

		# align camera
		self.camera_head_signal.match_scale_origin_2d(self.config['cam_scale_origin'])
		# discard before clap regions
		clap_range = self.calib_timecode['camera']['clap']
		start_timecode = clap_range[0] if self.include_clap else clap_range[-1]
		self.camera_head_signal.set_data_for_network(start_timecode=start_timecode)

	def align(self, is_called_last=False):
		if self.camera_head_signal.empty:
			return 
		# align_imu with cam mocap timecode/range
		# start_timecode, frame_num = self.mocap_data.get_time_range()
		start_timecode, end_timecode, frame_num = self.camera_head_signal.get_time_range()
		for i, joint_key in enumerate(self.watch_signal):
			if self.watch_signal[joint_key] is None:
				continue
			# resample time 
			self.watch_signal[joint_key].parse_and_resample2(start_timecode=start_timecode, end_timecode=end_timecode, frame_num=frame_num, gt_timecode_list=self.camera_head_signal.mocap_timecode)
			self.watch_signal[joint_key].imu_ori_to_joint()

			# offset
			if is_called_last:
				print("start adding manual offset")
				if 'imu_offset' in self.config:
					if i == self.config['imu_offset']['idx']:
						offset = conversions.A2R(np.array(self.config['imu_offset']['offset']))
						self.watch_signal[joint_key].imu_ori = self.watch_signal[joint_key].imu_ori @ offset

	
	def save_signal_info(self):
		self.save_cam_signal_info()
		self.save_imu_signal_info()

	# save imu info
	def save_imu_signal_info(self):
		input_ = {}
		# imu signals
		input_['imu_rot'] = []
		input_['imu_acc'] = []
		
		for i, joint in enumerate(self.wrist_joints):
			input_['imu_rot'].append(self.watch_signal[joint].imu_ori)
			input_['imu_acc'].append(self.watch_signal[joint].imu_acc)
		
		input_['imu_rot'] = np.stack(input_['imu_rot'], axis=1)
		input_['imu_acc'] = np.stack(input_['imu_acc'], axis=1)

		pkl_filepath = f"{self.data_folder_path}/imu_signal.pkl"
		pickle.dump(input_, open(pkl_filepath, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
		print("Done saving IMU signal info!")

	# save cam info
	def save_cam_signal_info(self):
		input_ = {}
		# camera signals
		input_['camera_signal'] = self.camera_head_signal
		
		pkl_filepath = f"{self.data_folder_path}/cam_signal.pkl"
		pickle.dump(input_, open(pkl_filepath, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
		print("Done saving cam info!")


	def set_mocap_input(self):
		self.head = self.camera_head_signal.mocap_cam2head

		frame_num, _, _ = self.head.shape
		self.imu_rot_input = np.zeros(shape=(frame_num, 2, 3, 3))
		self.imu_acc_input = np.zeros(shape=(frame_num, 2, 3))

		for i, joint in enumerate(self.wrist_joints):
			self.imu_rot_input[:,i,...] = self.watch_signal[joint].imu_ori
			self.imu_acc_input[:,i,...] = self.watch_signal[joint].imu_acc


	def sync_among_sensors(self):
		clap_range = self.calib_timecode['camera']['clap']
		# cam 
		clap_start_frame = time_to_frame(self.camera_head_signal.timecode, clap_range[0])
		self.clap_peak_cam = self.config['clap']['camera'] - clap_start_frame
		if self.manual_sync:
			return 
		clap_peak = {}
		for joint in self.wrist_joints:
			clap_frames_per_joint = time_range_into_frames(self.watch_signal[joint].timecode, clap_range[0], clap_range[-1])
			peak = self.watch_signal[joint].get_mid_peak_from_clap_range(clap_frames_per_joint)
			# clap_peak[joint] = peak 
			time_diff = self.clap_peak_cam - peak 
			self.watch_signal[joint].update_timecode(time_diff)

		# self.clap_peak_cam = clap_peak_cam

	def load(self):
		self.align()
		# if self.manual_sync is False:
		self.sync_among_sensors()
		# self.align(is_called_last=True) # realign with updated timecode

	# generate network input
	def preprocess_into_pkl(self, file_suffix=""):
		
		# get head and imu input from camera and imu
		self.set_mocap_input()

		# constants
		window = motion_constants.preprocess_window
		offset = 25

		head_T = []
		imu_rot = []
		imu_acc = []

		length, _, _ = self.head.shape
		i = 0
		while True:
			if i+window > length:
				break
			else:
				head_T_window = self.head[i:i+window]
				imu_rot_window = self.imu_rot_input[i:i+window]
				imu_acc_window = self.imu_acc_input[i:i+window]

			# record
			head_T.append(head_T_window)
			imu_rot.append(imu_rot_window)
			imu_acc.append(imu_acc_window)
			i += offset 
		
		head_T = np.asarray(head_T).astype(dtype=np.float32)
		imu_rot = np.asarray(imu_rot).astype(dtype=np.float32)
		imu_acc = np.asarray(imu_acc).astype(dtype=np.float32)
		
		# get head up vector before normalization
		height_indice = 1 if motion_constants.UP_AXIS == "y" else 2
		upvec_axis = np.array([0,0,0]).astype(dtype=np.float32)
		upvec_axis[1] = 1.0

		# get head up vector before normalization (here apply z!)
		head_upvec = np.einsum('ijkl,l->ijk', head_T[...,:3,:3], upvec_axis) # fixed bug! 
		head_height = head_T[...,height_indice,3][..., np.newaxis]

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

		# convert to dict
		input_ = {}
		input_['input_seq'] = head_imu_input
		input_['head_start'] = head_start_T[:,:,np.newaxis,...]
		input_['seq_len'] = window
		input_['offset'] = offset

		# save dict into pkl
		pkl_filepath = f"{self.data_folder_path}/{file_suffix}.pkl"
		pickle.dump(input_, open(pkl_filepath, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
		print(f"Done saving {file_suffix}")



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data-name", type=str, help="data name")
	parser.add_argument("--mocap", type=str, help="data name")


	args = parser.parse_args()
	rdc = RealDataCollection2(args.data_name)

	rdc.load()

	# get peaks for syncing 
	# rdc.sync_among_sensors()

	# draw specific range
	clap_range = rdc.calib_timecode['camera']['clap']
	clap_acc = {}    
	
	for joint in rdc.wrist_joints:
		clap_frames_per_joint = time_range_into_frames(rdc.watch_signal[joint].timecode, clap_range[0], clap_range[-1])
		print(f"start: {rdc.watch_signal[joint].timecode[clap_frames_per_joint[0]]} joint: {joint}")
		clap_acc[joint] = rdc.watch_signal[joint].user_acc_calibrated[clap_frames_per_joint[0]:clap_frames_per_joint[-1]+10]

	clap_acc_resampled = {}
	# clap_range_frame = rdc.camera_head_signal.mocap_frames[-1] - rdc.camera_head_signal.mocap_frames[0]

	# for joint in rdc.wrist_joints:
	# 	clap_acc_resampled[joint] = rdc.watch_signal[joint].imu_acc[:300]

	# plot_3_2side(clap_acc['LeftHand'], clap_acc['RightHand'], clap_acc_resampled['LeftHand'], clap_acc_resampled['RightHand'], x_vline_list=[57, 230, 82, 257])
	plot_3_2side(clap_acc['LeftHand'], clap_acc['RightHand'], clap_acc['RightHand'], x_vline_list=[rdc.clap_peak_cam])
	# plot_3_2side(clap_acc['LeftHand'], clap_acc['RightHand'], clap_acc['RightHand'])
