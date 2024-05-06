import sys, os
# sys.path.insert(0, os.path.dirname(__file__))
# sys.path.append("..")
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
from constants import motion_data as motion_constants
import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from fairmotion.core import motion as motion_class
from fairmotion.ops import conversions, motion as motion_ops
from fairmotion.utils import utils
from IPython import embed
from copy import deepcopy
import constants.motion_data as motion_constants
from fairmotion.core import motion as motion_classes
from fairmotion.ops import conversions, math as fairmotion_math
from fairmotion.data import bvh
import constants.motion_data as motion_constants
import imu2body.amass as amass

# from bvh import Skeleton

"""
Structure of npz file in AMASS dataset is as follows.
- trans (num_frames, 3):  translation (x, y, z) of root joint
- gender str: Gender of actor
- mocap_framerate int: Framerate in Hz
- betas (16): Shape parameters of body. See https://smpl.is.tue.mpg.de/
- dmpls (num_frames, 8): DMPL parameters
- poses (num_frames, 156): Pose data. Each pose is represented as 156-sized
	array. The mapping of indices encoding data is as follows:
	0-2 Root orientation
	3-65 Body joint orientations
	66-155 Finger articulations
"""

# assume beta is none
def bvh_to_amass_motion(bvh_filename, amass_skel):
	# load info
	bvh_motion = bvh.load(bvh_filename, scale=0.01)
	# body_model = amass.load_body_model(bm_path=motion_constants.BM_PATH)
	# skel_with_offset = amass.create_skeleton_from_amass_bodymodel(bm=body_model)	
	# amass_skel = skel_with_offset[0]
	# amass_skel_offset = skel_with_offset[1]

	amass_skel_ = deepcopy(amass_skel)

	# return bvh_motion
	# first rotate to z-up axis
	bvh_motion = motion_ops.rotate(bvh_motion, conversions.Ax2R(conversions.deg2rad(90)))

	# different joint ordering 
	bvh_to_matrix = bvh_motion.to_matrix()
	amass_matrix = np.zeros_like(bvh_to_matrix)
	frame = bvh_to_matrix.shape[0]

	# print("start")
	for joint in bvh_motion.skel.joints:
		joint_idx = bvh_motion.skel.get_index_joint(joint.name)
		amass_joint_idx = amass_skel_.get_index_joint(joint.name)
		# pose_matrix_amass[amass_joint_idx] = pose_matrix[joint_idx]
		amass_matrix[:,amass_joint_idx, ...] = bvh_to_matrix[:, joint_idx, ...]

	amass_motion = motion_classes.Motion.from_matrix(amass_matrix, skel=deepcopy(amass_skel_))
	amass_motion.set_fps(bvh_motion.fps)

	fps = bvh_motion.fps 
	if fps % motion_constants.FPS == 0:
		stride = int(fps / motion_constants.FPS)
		motion_resampled = amass_motion
		motion_resampled.poses = amass_motion.poses[::stride]
		motion_resampled.fps = motion_constants.FPS 
	elif fps != motion_constants.FPS:
		motion_resampled = motion_ops.resample(amass_motion, fps=motion_constants.FPS)
	elif fps == motion_constants.FPS:
		motion_resampled = amass_motion
	motion_adjust_height, foot_offset = motion_ops.adjust_height(motion_resampled, height_axis=motion_constants.UP_AXIS, pivot=motion_constants.contact_pivot)
	return motion_adjust_height

	# print(f"start resample of: {bvh_filename}")
	# motion_resampled = motion_ops.resample(amass_motion, fps=motion_constants.FPS) if amass_motion.fps != motion_constants.FPS else amass_motion 
	
	return motion_resampled

def create_skeleton_from_amass_bodymodel(bm, betas=None):
	num_joints = motion_constants.NUM_JOINTS
	joint_names = motion_constants.JOINT_NAMES

	pose_body_zeros = torch.zeros((1, 3 * (num_joints - 1))) # generate t-pose
	body = bm(pose_body=pose_body_zeros, betas=betas)
	base_position = body.Jtr.detach().numpy()[0, 0:num_joints]
	parents = bm.kintree_table[0].long()[:num_joints]
	joints = []
	for i in range(num_joints):
		joint = motion_class.Joint(name=joint_names[i])
		if i == 0:
			joint.info["dof"] = 6
			joint.xform_from_parent_joint = conversions.p2T(np.zeros(3))
		else:
			joint.info["dof"] = 3
			joint.xform_from_parent_joint = conversions.p2T(
				base_position[i] - base_position[parents[i]]
			)
		joints.append(joint)

	parent_joints = []
	for i in range(num_joints):
		parent_joint = None if parents[i] < 0 else joints[parents[i]]
		parent_joints.append(parent_joint)

	skel = motion_class.Skeleton(
		v_up=np.array([0.0, 1.0, 0.0]),
        v_face=np.array([0.0, 0.0, 1.0]),
        v_up_env=np.array([0.0, 0.0, 1.0]),
	)
	for i in range(num_joints):
		skel.add_joint(joints[i], parent_joints[i])

	return skel, base_position[0]


def create_motion_from_amass_data(filename, bm, skel_with_offset=None, default_beta=True, load_motion=True):        
	try:
		bdata = np.load(filename)
	except:
		print(f"Error in loading: {filename}. Return None")
		return None
	betas = None if default_beta else torch.Tensor(bdata["betas"][np.newaxis]).to("cpu")

	if skel_with_offset is None:
		skel, offset = create_skeleton_from_amass_bodymodel(
			bm, betas=betas
		)
	else: 
		skel, offset = deepcopy(skel_with_offset)
	
	if "mocap_frame_rate" not in bdata.files:
		return None
	fps = float(bdata["mocap_frame_rate"])    
	root_orient = bdata["poses"][:, :3]  # controls the global root orientation (frame, 3)
	pose_body = bdata["poses"][:, 3:66]  # controls body joint angles (frame, 63)
	trans = bdata["trans"][:, :3]  # controls root translation (frame, 3)

	motion = motion_class.Motion(skel=skel, fps=fps)

	if not load_motion:
		return motion 
	
	num_joints = skel.num_joints()
	parents = bm.kintree_table[0].long()[:num_joints]

	for frame in range(pose_body.shape[0]):
		pose_body_frame = pose_body[frame]
		root_orient_frame = root_orient[frame]
		root_trans_frame = trans[frame] + offset # add offset 
		pose_data = []
		for j in range(num_joints):
			if j == 0:
				T = conversions.Rp2T(
					conversions.A2R(root_orient_frame), root_trans_frame
				)
			else:
				T = conversions.R2T(
					conversions.A2R(
						pose_body_frame[(j - 1) * 3 : (j - 1) * 3 + 3]
					)
				)
			pose_data.append(T)
		motion.add_one_frame(pose_data)
	
	if motion_constants.UP_AXIS == "y":
		motion = motion_ops.rotate(motion, conversions.Ax2R(conversions.deg2rad(-90))) # match to y up axis

	if fps % motion_constants.FPS == 0:
		stride = int(fps / motion_constants.FPS)
		motion_resampled = motion
		motion_resampled.poses = motion.poses[::stride]
		motion_resampled.fps = motion_constants.FPS 
	elif fps != motion_constants.FPS:
		motion_resampled = motion_ops.resample(motion, fps=motion_constants.FPS)
	elif fps == motion_constants.FPS:
		motion_resampled = motion
	motion_adjust_height, foot_offset = motion_ops.adjust_height(motion_resampled, height_axis=motion_constants.UP_AXIS, pivot=motion_constants.contact_pivot)
	return motion_adjust_height


	# motion_resampled = motion_ops.resample(motion, fps=motion_constants.FPS) if fps != motion_constants.FPS else motion 
	# motion_adjust_height, foot_offset = motion_ops.adjust_height(motion_resampled, height_axis=motion_constants.UP_AXIS, pivot=motion_constants.contact_pivot)
	# return motion_adjust_height

def create_mesh_from_amass_fairmotion(motion, bm, offset=None, betas=None):

	if offset is None: # offset with default betas
		offset = np.array([ 0.00312326, -0.35140747,  0.01203655])

	motion_T = motion.to_matrix(local=True)

	trans = motion_T[:,0,:3,3] - offset[np.newaxis, ...]
	joint_aa = conversions.R2A(conversions.T2R(motion_T))

	poses = joint_aa[:,1:,...]
	poses = poses.reshape(-1, 63)
	root_orient = joint_aa[:,0,...]

	# add default hand pose
	frames, dof = poses.shape 
	default_pose_hand = motion_constants.pose_hand
	default_pose_hand = default_pose_hand.unsqueeze(0).repeat(frames, 1)

	# change to tensor
	trans = torch.from_numpy(trans).float()
	poses = torch.from_numpy(poses).float()
	root_orient = torch.from_numpy(root_orient).float()
	
	# send to bm
	body = bm(pose_body=poses, root_orient=root_orient, pose_hand=default_pose_hand, betas=betas, trans=trans)
	vertices_seq = body.v.numpy()
	faces = body.f.numpy()

	return vertices_seq, faces 


# deprecated
def create_mesh_from_amass_data(filename, bm, default_beta=True):
	bdata = np.load(filename)
	betas = None if default_beta else torch.from_numpy(bdata["betas"][np.newaxis]).to("cpu")     
	fps = float(bdata["mocap_frame_rate"])    
	stride = 2 if fps == 120 else 1
	assert fps in [60, 120], "fps should either be 60 or 120!" # TODO add interpolation via motion

	trans = torch.from_numpy(bdata['trans'][::stride,:3]).float()
	poses = torch.from_numpy(bdata["poses"][::stride,3:66]).float()
	# hand 
	pose_hand = torch.from_numpy(bdata["poses"][::stride,75:75+90]).float() # index for smplx
	root_orient = torch.from_numpy(bdata["poses"][::stride,:3]).float()
	frames, dof = poses.shape
	body = bm(pose_body=poses, root_orient=root_orient, pose_hand=pose_hand, betas=betas, trans=trans)
	vertices_seq = body.v.numpy()
	faces = body.f.numpy()
	frame, num_vertice, _ = vertices_seq.shape
	mesh_seq = []
	import trimesh
	for i in range(frame):
		mesh = trimesh.Trimesh(vertices=vertices_seq[i], faces=faces, force_mesh=True, process=False)
		rotate = conversions.Ax2R(conversions.deg2rad(-90)) # match to y up axis
		mesh.apply_transform(conversions.R2T(rotate))
		mesh_seq.append(mesh)
	return mesh_seq
	

def load_body_model(bm_path):
	comp_device = torch.device("cpu")
	bm = BodyModel(
		bm_fname=bm_path, 
		num_betas=motion_constants.BETA_DIM, 
	).to(comp_device)
	return bm


def create_tpose(bm_path, default_beta=True):
	bm = load_body_model(bm_path)

	betas = None 

	skel, offset = create_skeleton_from_amass_bodymodel(
		bm, betas=betas
	)
		
	# root_orient = bdata["poses"][:, :3]  # controls the global root orientation (frame, 3)
	# pose_body = bdata["poses"][:, 3:66]  # controls body joint angles (frame, 63)
	# trans = bdata["trans"][:, :3]  # controls root translation (frame, 3)
	
	root_orient = np.zeros(shape=(1,3))
	pose_body = np.zeros(shape=(1,63))
	trans = np.zeros(shape=(1,3))

	root_orient_tensor = torch.from_numpy(root_orient).float()
	pose_body_tensor = torch.from_numpy(pose_body).float()
	trans_tensor = torch.from_numpy(trans).float()

	frames = 1
	dof = 63

	body = bm(pose_body=pose_body_tensor, root_orient=root_orient_tensor, betas=betas, trans=trans_tensor)
	vertices_seq = body.v.numpy()
	faces = body.f.numpy()
	frame, num_vertice, _ = vertices_seq.shape
	mesh_seq = []

	import trimesh
	for i in range(frame):
		mesh = trimesh.Trimesh(vertices=vertices_seq[i], faces=faces, force_mesh=True, process=False)
		# rotate = conversions.Ax2R(conversions.deg2rad(90)) # match to z up axis
		# mesh.apply_transform(conversions.R2T(rotate))
		mesh_seq.append(mesh)

	motion = motion_class.Motion(skel=skel, fps=30)

	# embed()

	# if not load_motion:
	#     return motion 
	
	num_joints = skel.num_joints()
	parents = bm.kintree_table[0].long()[:num_joints]

	for frame in range(pose_body.shape[0]):
		pose_body_frame = pose_body[frame]
		root_orient_frame = root_orient[frame]
		root_trans_frame = trans[frame] + offset # add offset 
		pose_data = []
		for j in range(num_joints):
			if j == 0:
				T = conversions.Rp2T(
					conversions.A2R(root_orient_frame), root_trans_frame
				)
			else:
				T = conversions.R2T(
					conversions.A2R(
						pose_body_frame[(j - 1) * 3 : (j - 1) * 3 + 3]
					)
				)
			pose_data.append(T)
		motion.add_one_frame(pose_data)

	# motion = motion_ops.rotate(motion, conversions.Ax2R(conversions.deg2rad(90))) # match to z up axis

	return motion, mesh_seq, offset
	# if motion_constants.UP_AXIS == "y":

	# motion_resampled = motion_ops.resample(motion, fps=motion_constants.FPS) if fps != motion_constants.FPS else motion 
	# motion_adjust_height, foot_offset = motion_ops.adjust_height(motion_resampled, height_axis=motion_constants.UP_AXIS)
	
	# return motion_adjust_height



def load(file, bm=None, bm_path=None, load_motion=True, load_mesh=False):
	if bm is None:
		assert bm_path is not None, "Please provide SMPL body model path"
		bm = load_body_model(bm_path)
	motion_sequence =  create_motion_from_amass_data(
		filename=file, bm=bm, load_motion=load_motion)
	mesh_sequence = None if not load_mesh else create_mesh_from_amass_data(file, bm=bm)
	return motion_sequence, mesh_sequence


def save():
	raise NotImplementedError("Using bvh.save() is recommended")


def load_parallel(files, cpus=20, **kwargs):
	return utils.run_parallel(load, files, num_cpus=cpus, **kwargs)


if __name__ == "__main__":
	bm_path = "../data/smpl_models/smplx/SMPLX_NEUTRAL.npz"
	bm = load_body_model(bm_path=bm_path)
	pose_body_zeros = torch.zeros((1, 3 * (motion_constants.NUM_JOINTS - 1))) # generate t-pose
	body = bm(pose_body=pose_body_zeros, betas=None)
	vertices_seq = body.v.numpy()
	faces = body.f.numpy()
	frame, num_vertice, _ = vertices_seq.shape
	mesh_seq = []

	import trimesh
	
	for i in range(frame):
		mesh = trimesh.Trimesh(vertices=vertices_seq[i], faces=faces, force_mesh=True, process=False)
		mesh.export(f"smpl_test{i}.obj")
	
	#     rotate = conversions.Ax2R(conversions.deg2rad(-90)) # match to y up axis
	#     mesh.apply_transform(conversions.R2T(rotate))
	#     mesh_seq.append(mesh)
	# return mesh_seq
