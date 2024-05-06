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
# from sensor2body.footcleanup import FootCleanup
import constants.motion_data as motion_constants
from autoencoder_v2.run import AERenderData 
import imu2body_eval.amass_smplh as amass_smplh
import imu2body.amass as amass 

translate_offset = np.array([0.5, 0.0,0.0])
CUR_BM_TYPE = "smplx"

def convert_motion(T, skel, translate_offset=None):

	motion = motion_class.Motion.from_matrix(T, deepcopy(skel))
	if translate_offset is not None:
		motion = motion_ops.translate(motion, translate_offset)
	motion.set_fps(motion_constants.FPS)
	return motion


def parse_output(args):
	# return dictionary
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
		description="Visualize testset results"
	)
	
	from visualizer.render_args import add_render_argparse
	parser.add_argument("--test-name")
	parser.add_argument("--idx")
	parser.add_argument("--file-name")
	
	parser = add_render_argparse(parser=parser)
	args = parser.parse_args()

	args.axis_up = motion_constants.UP_AXIS	
	result = parse_output(args)
	
	vis = visualizer.load_visualizer(result_dict=result, args=args)

	vis.run() 
