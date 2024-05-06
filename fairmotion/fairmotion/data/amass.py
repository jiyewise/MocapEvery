# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from fairmotion.core import motion as motion_class
from fairmotion.ops import conversions, motion as motion_ops
from fairmotion.utils import utils
from IPython import embed
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

""" 
From frankmocap docs
0: Global
1: L_Hip
2: R_Hip
3: Spine_01
4: L_Knee
5: R_Knee
6: Spine_02
7: L_Ankle
8: R_Ankle
9: Spine_03
10: L_Toe
11: R_Toe
12: Neck
13: L_Collar
14: R_Collar
15: Head
16: L_Shoulder
17: R_Shoulder
18: L_Elbow
19: R_Elbow
20: L_Wrist
21: R_Wrist
22: L_Palm (Invalid for SMPL-X/SMPL-H)
23: R_Palm (Invalid for SMPL-X/SMPL-H)
"""

# Custom names for 22 joints in AMASS data
joint_names = [
    "Hips",
    "LeftUpLeg",
    "RightUpLeg",
    "Spine",
    "LeftLeg",
    "RightLeg",
    "Spine1",
    "LeftFoot",
    "RightFoot",
    "Spine2",
    "LeftToe",
    "RightToe",
    "Neck",
    "LeftShoulder",
    "RightShoulder",
    "Head",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand"
]

# Custom names for 22 joints in AMASS data
# joint_names = [
#     "root",
#     "lhip",
#     "rhip",
#     "lowerback",
#     "lknee",
#     "rknee",
#     "upperback",
#     "lankle",
#     "rankle",
#     "chest",
#     "ltoe",
#     "rtoe",
#     "lowerneck",
#     "lclavicle",
#     "rclavicle",
#     "upperneck",
#     "lshoulder",
#     "rshoulder",
#     "lelbow",
#     "relbow",
#     "lwrist",
#     "rwrist",
# ]


def create_skeleton_from_amass_bodymodel(bm, num_joints, joint_names, betas=None):
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

    skel = motion_class.Skeleton()
    for i in range(num_joints):
        skel.add_joint(joints[i], parent_joints[i])

    return skel, base_position[0]



def create_motion_from_amass_data(filename, bm, skel_with_offset=None, fix_betas=True, load_motion=True):
    bdata = np.load(filename)
    betas = None if fix_betas else torch.Tensor(bdata["betas"][np.newaxis]).to("cpu")

    if skel_with_offset is None:
        skel, offset = create_skeleton_from_amass_bodymodel(
            bm, len(joint_names), joint_names, betas=betas
        )
    else: 
        skel, offset = skel_with_offset
    
    if "mocap_frame_rate" not in bdata.files:
        return None
    fps = float(bdata["mocap_frame_rate"])    
    stride = 2 if fps == 120 else 1
    # print(fps)
    assert fps in [60, 120], "fps should either be 60 or 120!"

    root_orient = bdata["poses"][::stride, :3]  # controls the global root orientation
    pose_body = bdata["poses"][::stride, 3:66]  # controls body joint angles
    trans = bdata["trans"][::stride, :3]  # controls root translation

    motion = motion_class.Motion(skel=skel, fps=60)

    if not load_motion:
        return motion 
    
    num_joints = skel.num_joints()
    parents = bm.kintree_table[0].long()[:num_joints]

    for frame in range(pose_body.shape[0]):
        pose_body_frame = pose_body[frame]
        root_orient_frame = root_orient[frame]
        # root_trans_frame = trans[frame] + np.array([-0.00217366, -0.24078917,  0.02858375]) # smplh
        # root_trans_frame = trans[frame] + np.array([ 0.00312326, -0.35140747,  0.01203655]) # smplx
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
    
    motion = motion_ops.rotate(motion, conversions.Ax2R(conversions.deg2rad(-90))) # match to y up axis
    return motion

def create_mesh_from_amass_data(filename, bm, fix_betas=True):
    num_joints = len(joint_names)
    bdata = np.load(filename)
    betas = None if fix_betas else torch.from_numpy(bdata["betas"][np.newaxis]).to("cpu")     
    fps = float(bdata["mocap_frame_rate"])    
    stride = 2 if fps == 120 else 1
    assert fps in [60, 120], "fps should either be 60 or 120!"

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
        mesh = trimesh.Trimesh(vertices=vertices_seq[i], faces=faces, force_mesh=True)
        rotate = conversions.Ax2R(conversions.deg2rad(-90)) # match to y up axis
        mesh.apply_transform(conversions.R2T(rotate))
        mesh_seq.append(mesh)
    return mesh_seq
    

def load_body_model(bm_path, num_betas=16):
    comp_device = torch.device("cpu")
    bm = BodyModel(
        bm_fname=bm_path, 
        num_betas=num_betas, 
        # model_type=model_type
    ).to(comp_device)
    # print("betas: ", num_betas)
    return bm


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
