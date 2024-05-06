import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
# import visualizer.rdn_scene_visualizer as vis
import argparse
from fairmotion.core import motion as motion_class
from fairmotion.ops import conversions, math, motion as motion_ops
from IPython import embed 
from realdata.realdata_collection import *
from realdata.realdata_network_run import *
import cv2
from realdata.footcleanup import * 

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Real Data Visualization"
	)

	parser.add_argument("--rdc-folder", type=str, default="", required=True)
	parser.add_argument("--test-name", type=str, default="")
	parser.add_argument("--bm-type",default="smplx")
	parser.add_argument("--demo", default=True)
	parser.add_argument("--cleanup", default=True) # footcleanup post-processing (only for real-data demos)
	parser.add_argument("--motion-only", action='store_true')

	from visualizer.render_args import add_render_argparse
	parser = add_render_argparse(parser=parser)

	args = parser.parse_args()
	args.axis_up = "z" # z
	args.axis_face = "x" # x

	result_dict = parse_realdata_network_output(args=args)
	
	# clean up pcd
	cl, ind = result_dict['pcd'].remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
	result_dict['pcd'] = result_dict['pcd'].select_by_index(ind)

	# minor adjustment not to collide with toe mesh (vis)
	result_dict['pcd'].translate((0, 0, -0.10))

	mocap_img_list = result_dict['head_signal'].mocap_img_list
	# load texture
	img_texture_list = utils.run_parallel(cv2.imread, mocap_img_list, num_cpus=24)
	
	result_dict['img_texture'] = img_texture_list

	if args.motion_only:
		import demo_video_visualizer.motion_visualizer as vis 
		result_dict['motion']['result'] = deepcopy(result_dict['motion']['watch_cam'])
		result_dict['motion']['result'] = motion_ops.translate(result_dict['motion']['result'], [0,0,0.05])
		result_dict['motion']['result'] = motion_ops.rotate(result_dict['motion']['result'], conversions.Ax2R(conversions.deg2rad(-90)),)
		del result_dict['motion']['watch_cam']

		args.up_axis = "y"
		args.axis_face = "z"
		network_vis = vis.load_visualizer(result_dict=result_dict, args=args)
		network_vis.run()

	if args.demo:
		import demo_video_visualizer.rdn_visualizer as vis 
		result_dict['motion']['result'] = deepcopy(result_dict['motion']['watch_cam'])
		del result_dict['motion']['watch_cam']

		network_vis = vis.load_visualizer(result_dict=result_dict, args=args)
		network_vis.run()

	network_vis = vis.load_visualizer(result_dict=result_dict, args=args)
	network_vis.run()

