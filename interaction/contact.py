from copy import deepcopy
from matplotlib.pyplot import axes
import torch
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import logging
import numpy as np
import os
import pickle

# from bvh import *
from fairmotion.core import motion as motion_classes
from fairmotion.ops import conversions, math as fairmotion_math
from fairmotion.data import bvh
import sys
from datetime import datetime
from copy import deepcopy
from imu2body.functions import *
import imu2body.amass as amass
from fairmotion.utils import utils
from tqdm import tqdm
from copy import deepcopy
import constants.imu as imu_constants
import constants.motion_data as motion_constants
import imu2body.imu as imu
from IPython import embed
from fairmotion.core import motion as motion_class

class ContactManagerCustomRun(object):
	def __init__(self, contact_dict) -> None:
		self.height_offset = 0.0
		self.contact_frame = 0
		self.contact_dict = contact_dict
		self.set_skel_info()

		# contact history 
		self.contact_history = None

		# pcd projection
		self.pcd_points = None
		self.projection_xy_dist_threshold = 0.15

		self.height_diff_threshold = 0.1

	def add_pcd_points(self, pcd_points):
		self.pcd_points = pcd_points

	def set_contact_dict(self, contact_dict):
		self.contact_dict = contact_dict

	def set_skel_info(self):
		body_model = amass.load_body_model(motion_constants.BM_PATH)
		fairmotion_skel, _ = amass.create_skeleton_from_amass_bodymodel(bm=body_model)

		self.skel = fairmotion_skel
		self.skel_offset = fairmotion_skel.get_joint_offset_list()
		self.skel_parent = fairmotion_skel.get_parent_index_list()
		self.ee_idx = fairmotion_skel.get_index_joint(motion_constants.EE_JOINTS)
		self.foot_idx = fairmotion_skel.get_index_joint(motion_constants.FOOT_JOINTS)
		self.hand_idx = fairmotion_skel.get_index_joint(motion_constants.HAND_JOINTS)
		# this is to solve overfitting on foot
		self.leg_idx = fairmotion_skel.get_index_joint(motion_constants.LEG_JOINTS)

		self.skel_offset = self.skel_offset[np.newaxis, np.newaxis, ...] 		# expand skel offset into tensor

	def check_need_update(self, rd):
		if self.contact_history is None:
			return True
		prev_left_contact = np.all(self.contact_history[-5:,0] > 0.5)
		prev_right_contact = np.all(self.contact_history[-5:,1] > 0.5)

		start_left_contact = np.all(rd.contact_label[-5:,0] > 0.5)
		start_right_contact = np.all(rd.contact_label[-5:,1] > 0.5)

		is_all_contact = all([prev_left_contact, prev_right_contact, start_left_contact, start_right_contact])
		
		if is_all_contact:
			return False
		return True

	def update(self, rd, frame_start):
		self.frame_start = frame_start
		self.update_contact_history(rd.contact_label)

		is_update = self.check_need_update(rd=rd)
		
		if not is_update:
			# print(f"Do not update at frame: {frame_start}!")
			return None
		updated_height_offset, updated_contact_frame = self.calc_contact_info(rd=rd, frame_start=frame_start)


		try:
			if updated_contact_frame > self.contact_frame:
				self.contact_frame = updated_contact_frame
				self.height_offset = updated_height_offset
				print(f"updated contact frame: {updated_contact_frame} height: {self.height_offset}")
				self.contact_dict[self.contact_frame] = self.height_offset
				return self.contact_frame
			else:
				return None
		except:
			embed()


	def update_contact_history(self, rd_contact_label):
		# save contact history 
		if self.contact_history is None:
			self.contact_history = rd_contact_label 
		else: 
			self.contact_history = np.concatenate([self.contact_history, rd_contact_label], axis=0)


	def calc_contact_info(self, rd, frame_start):
		# consider past contact history and decide to update or not
		
		# TODO later change to torch tensor or so.. (to be faster)
		use_contact_foot_pos = True if self.pcd_points is not None else False
		T = rd.output_T
		contact_label = rd.contact_label
		# embed()

		motion = motion_class.Motion.from_matrix(T, deepcopy(self.skel))
		result_dict = update_height_offset(motion.to_matrix(local=False), prev_offset=self.height_offset, frame_start=frame_start, return_contact_foot_pos=use_contact_foot_pos, contact_labels=contact_label)

		updated_height_offset = result_dict['height']
		updated_contact_frame = result_dict['frame']
		contact_foot_pos  = result_dict['contact_pos']
		
		if contact_foot_pos is not None:
			projected_height_offset = self.project_to_pcd(contact_foot_pos=contact_foot_pos)
			if projected_height_offset is not None:
				# print(f"found projection in frame: {updated_contact_frame}")
				updated_height_offset = projected_height_offset
				if abs(updated_height_offset - self.height_offset) < self.height_diff_threshold: # handle noise
					updated_height_offset = self.height_offset
					updated_contact_frame = -1

			# if projected_height_offset is None:
			# 	updated_height_offset = self.height_offset
			# 	updated_contact_frame = -1


		# ignore noise
		# if contact_foot_pos is None:
		if abs(updated_height_offset - self.height_offset) < self.height_diff_threshold:
			updated_height_offset = self.height_offset
			updated_contact_frame = -1
		
		return updated_height_offset, updated_contact_frame

	def parse_contact_history(self, num_frames_to_remove):
		self.contact_history = self.contact_history[:-num_frames_to_remove]

	def project_to_pcd(self, contact_foot_pos):
		if self.pcd_points is None:
			return None 
		
		# try:
		# print(f"contact pos: {contact_foot_pos}")
		xy_distances = np.linalg.norm(self.pcd_points[:, :2] - contact_foot_pos[:2], axis=1)
		# except:
		# 	embed()
		# Filter out points within the distance_threshold in the XY plane
		filtered_points = self.pcd_points[xy_distances < self.projection_xy_dist_threshold]
		# print(f" number of close points: {len(filtered_points)}")
		# If no points found within threshold, return None
		if len(filtered_points) == 0:
			# print("no close points")
			return None

		# z_diffs = np.abs(filtered_points[:, 2] - contact_foot_pos[2])
		# further_filtered_points = filtered_points[(z_diffs >= 0) & (z_diffs <= 0.2)]

		# if len(further_filtered_points):
		# 	print("no close points 2")
		# 	return None 
		# Find the index of the point with the closest Z value to the reference point
		# closest_z_index = np.argmin(np.abs(filtered_points[:, 2] - contact_foot_pos[2]))
		z_diff_min = 0.0
		z_diff_max = 0.1

		z_diffs = np.abs(filtered_points[:, 2] - contact_foot_pos[2])
		further_filtered_points = filtered_points[(z_diffs >= z_diff_min) & (z_diffs <= z_diff_max)]

		# If no points found within criteria, return None
		if len(further_filtered_points) < 10:
			# print("no close points 2")
			return None

		# Calculate the average of the further filtered points
		avg_point = np.mean(further_filtered_points, axis=0)
		return avg_point[2]
		# return None

	def fk(self):
		return 	


def get_height_offset_current_frame(contact_dict, cur_frame):
    keys_below_threshold = [key for key in contact_dict if key <= cur_frame]
    if keys_below_threshold:
        max_key = max(keys_below_threshold)
        return max_key, contact_dict[max_key]
    return 0, 0.0

def generate_contact_labels(global_T, only_foot=True): # move contact related config to motion constants
	if only_foot:
		contact_joint_idx = motion_constants.foot_joint_idx # NOTE use toe? foot?

	contact_vel = (global_T[1:,contact_joint_idx,:3,3] - global_T[:-1,contact_joint_idx,:3,3])
	contact_vel_norm = np.linalg.norm(contact_vel, axis=2)
	contact = contact_vel_norm < motion_constants.contact_vel_threshold
	contact = np.concatenate((contact[:,0:1], contact), axis=-1) # pad first frame
	return contact 

def update_height_offset(global_T, prev_offset, frame_start, return_contact_labels=False, return_contact_foot_pos=False, contact_labels=None):
	if contact_labels is not None:
		foot_contact = contact_labels
	else:
		foot_contact = generate_contact_labels(global_T=global_T, only_foot=True)

	toe_joint_idx = motion_constants.toe_joints_idx
	height_indice = 1 if motion_constants.UP_AXIS == "y" else 2

	contact_frames_left = np.where(foot_contact[:,0])[0]
	contact_frames_right = np.where(foot_contact[:,1])[0]

	new_contact_height = prev_offset
	update_contact_frame = -1

	contact_foot_pos = None

	# no contact for both
	# if len(contact_frames_left) == 0 and len(contact_frames_right) == 0:
	# 	return new_contact_height, update_contact_frame, foot_contact
	
	# only right
	if len(contact_frames_left) == 0 and len(contact_frames_right) > 0:
		right_contact_height = global_T[contact_frames_right, toe_joint_idx[1], height_indice, 3]
		new_contact_height = right_contact_height[-1]
		update_contact_frame = frame_start + contact_frames_right[-1]
		contact_foot_pos = global_T[contact_frames_right[-1], toe_joint_idx[1], :3, 3]

	# only left
	if len(contact_frames_right) == 0 and len(contact_frames_left) > 0:	
		left_contact_height = global_T[contact_frames_left, toe_joint_idx[0], height_indice, 3]
		new_contact_height = left_contact_height[-1]
		update_contact_frame = frame_start + contact_frames_left[-1]
		contact_foot_pos = global_T[contact_frames_left[-1], toe_joint_idx[0], :3, 3]

	# both: how to get?
	elif len(contact_frames_left) > 0 and len(contact_frames_right) > 0:
		last_contact_frame_left = contact_frames_left[-1]
		last_contact_frame_right = contact_frames_right[-1]
		lr = 0 if last_contact_frame_left > last_contact_frame_right else 1			
		last_frame = max(last_contact_frame_left, last_contact_frame_right)

		new_contact_height = global_T[last_frame, toe_joint_idx[lr], height_indice, 3]
		update_contact_frame = frame_start + last_frame
		contact_foot_pos = global_T[last_frame, toe_joint_idx[lr], :3, 3]
	
	# ignore noise
	# if abs(new_contact_height - prev_offset) < 0.1:
	# 	new_contact_height = prev_offset
	# 	update_contact_frame = -1
	# 	contact_foot_pos = None
		# return prev_offset, -1, foot_contact
	
	return_foot_contact = None if not return_contact_labels else foot_contact
	return_contact_foot_pos_result = None if not return_contact_foot_pos else contact_foot_pos

	return_dict = {}
	return_dict['height'] = new_contact_height
	return_dict['frame'] = update_contact_frame
	return_dict['contact_label'] = return_foot_contact
	return_dict['contact_pos'] = return_contact_foot_pos_result

	return return_dict