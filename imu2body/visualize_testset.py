import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
import numpy as np
import visualizer.imu2body_testset_visualizer as visualizer
import imu2body.amass as amass
import argparse
import pickle
from fairmotion.core import motion as motion_class
from fairmotion.ops import conversions, math, motion as motion_ops
from copy import deepcopy
from IPython import embed 
import random
from tqdm import tqdm
from fairmotion.utils import utils
from imu2body.functions import * 
import constants.motion_data as motion_constants
import imu2body_eval.amass_smplh as amass_smplh

CUR_BM_TYPE = "smplx"

class RenderData(object):
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


	def set_other_pos_estimation(self, other_est):
		self.other_est = {}
		for est_key in other_est:
			seq_len, dim  = other_est[est_key].shape
			est = other_est[est_key].reshape(seq_len, 2, -1)
			if dim > 3:
				est = est[...,-3:]
			est_T = conversions.p2T(est)

			self.other_est[est_key] = conversions.T2p(self.start_T @ est_T)

	def set_foot_estimation(self, foot_est):
		if self.start_T is not None:
			# adjust foot estimated to original
			foot_est = self.start_T @ conversions.p2T(foot_est)
			foot_est = conversions.T2p(foot_est)
		self.foot_est = foot_est


translate_offset = np.array([0.5, 0.0,0.0])

def convert_motion(T, skel, translate_offset=None):
	motion = motion_class.Motion.from_matrix(T, deepcopy(skel))
	if translate_offset is not None:
		motion = motion_ops.translate(motion, translate_offset)
	motion.set_fps(motion_constants.FPS)
	return motion

def parse_output(args):
	result_dict = {}

	# select random 200 indices and convert to fairmotion class
	num_motions = 200
	filepath = f"./output/{args.test_name}/testset_{args.idx}/{args.file_name}.pkl"
	data = pickle.load(open(filepath, "rb"))

	selected_idx = random.sample(range(0, len(data['output'])), num_motions)
	result_dict['idx'] = selected_idx
	output_array =  [data['output'][idx] for idx in selected_idx]
	gt_array =  [data['gt'][idx] for idx in selected_idx]

	if CUR_BM_TYPE == "smplx":
		body_model = amass.load_body_model(motion_constants.BM_PATH) 
		fairmotion_skel, _ = amass.create_skeleton_from_amass_bodymodel(bm=body_model)

	elif CUR_BM_TYPE == "smplh":
		body_model = amass_smplh.load_body_model(motion_constants.SMPLH_BM_PATH) 
		fairmotion_skel, _ = amass_smplh.create_skeleton_from_amass_bodymodel(bm=body_model)

	output_motion_list = utils.run_parallel(convert_motion, output_array, num_cpus=20, skel=fairmotion_skel, translate_offset=translate_offset)
	gt_motion_list = utils.run_parallel(convert_motion, gt_array, num_cpus=20, skel=fairmotion_skel)

	result_dict['motion'] = [[gt_motion_list[i], output_motion_list[i]] for i in range(num_motions)]

	result_dict['fps'] = motion_constants.FPS
	result_dict['seq_len'] = motion_constants.preprocess_window
	result_dict['translate_offset'] = translate_offset

	return result_dict


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Visualize imu22body testset results"
	)
	
	from visualizer.render_args import add_render_argparse
	parser.add_argument("--test-name")
	parser.add_argument("--idx")
	parser.add_argument("--file-name")
	parser.add_argument("--skel", type=str)
	parser.add_argument("--bm-path",default="../data/smpl_models/smplx/SMPLX_NEUTRAL.npz")\
	
	parser = add_render_argparse(parser=parser)
	args = parser.parse_args()

	args.axis_up = motion_constants.UP_AXIS	
	result = parse_output(args)
	
	vis = visualizer.load_visualizer(result_dict=result, args=args)

	vis.run() 
