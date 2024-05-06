import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
import argparse
from fairmotion.core import motion as motion_class
from fairmotion.ops import conversions, math, motion as motion_ops
from IPython import embed 
from realdata.realdata_collection import *
from realdata.realdata_network_run import *
import constants.motion_data as motion_constants
import constants.path as path_constants
from imu2body.imu import *
from autoencoder_v2.run import * 
from interaction.autoencoder_optimizer import * 
from fairmotion.ops.math import * 
from visualizer.render_args import add_render_argparse
from interaction.editor import postprocess_edit_result


parser = argparse.ArgumentParser(
description="Real Data Visualization"
)

parser.add_argument("--save-name", type=str, default="")
parser.add_argument("--bm-type", type=str, default="smplx")

parser = add_render_argparse(parser=parser)

args = parser.parse_args()
args.axis_up = "z" # z
args.axis_face = "x" # x

result_dict = postprocess_edit_result(args.save_name, draw_reproj=True)

bm_path = motion_constants.BM_PATH if args.bm_type == "smplx" else motion_constants.SMPLH_BM_PATH
body_model = amass.load_body_model(bm_path=bm_path)
amass_skel, offset = amass.create_skeleton_from_amass_bodymodel(bm=body_model, \
																betas=None 
															)


import demo_video_visualizer.edit_visualizer as demo_vis
# clean up pcd
cl, ind = result_dict['pcd'].remove_statistical_outlier(nb_neighbors=60, std_ratio=1.5)
result_dict['pcd'] = result_dict['pcd'].select_by_index(ind)

# adjustment for toe mesh (vis)
result_dict['pcd'].translate((0, 0, -0.05))

demo_editor_vis = demo_vis.load_visualizer(result_dict=result_dict, args=args)

demo_editor_vis.run()