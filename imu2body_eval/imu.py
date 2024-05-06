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
import constants.imu as imu_constants
import math
from scipy import ndimage 

imu_smooth_window = 3
average_windows = imu_smooth_window * 2 + 1

# reference: TIP (2022SA)
def _syn_acc(pos):

    num_frame, num_joint, _ = pos.shape 
    acc = np.zeros((num_frame, num_joint, 3), float)

    for t in range(imu_constants.syn_frame_n, num_frame - imu_constants.syn_frame_n):
        next_pos = pos[t+imu_constants.syn_frame_n]
        prev_pos = pos[t-imu_constants.syn_frame_n]
        cur_pos = pos[t]

        acc_t = (prev_pos + next_pos - 2*cur_pos) / math.pow(imu_constants.syn_acc_dt, 2)
        acc[t] = acc_t

    # pad boundaries 
    try:
        acc[:imu_constants.syn_frame_n] = acc[imu_constants.syn_frame_n]
        acc[-imu_constants.syn_frame_n:] = acc[-imu_constants.syn_frame_n-1]
    except:
        embed()
    return acc

def _syn_acc_torch(pos, device="cpu"):

    num_frame, num_joint, _ = pos.shape 
    acc = torch.zeros((num_frame, num_joint, 3)).to(device).float()

    for t in range(imu_constants.syn_frame_n, num_frame - imu_constants.syn_frame_n):
        next_pos = pos[t+imu_constants.syn_frame_n]
        prev_pos = pos[t-imu_constants.syn_frame_n]
        cur_pos = pos[t]

        acc_t = (prev_pos + next_pos - 2*cur_pos) / math.pow(imu_constants.syn_acc_dt, 2)
        acc[t] = acc_t

    # pad boundaries 
    try:
        acc[:imu_constants.syn_frame_n] = acc[imu_constants.syn_frame_n]
        acc[-imu_constants.syn_frame_n:] = acc[-imu_constants.syn_frame_n-1]
    except:
        embed()
    return acc


def imu_from_global_T(global_T, joint_idx=None):
    if joint_idx is None:
        joint_idx = motion_constants.imu_hand_joint_idx
    joint_global_rot = global_T[...,joint_idx,:3,:3]
    global_pos = global_T[...,:3,3]
    joint_global_pos = global_pos[...,joint_idx,:]    
    joint_acc = _syn_acc(joint_global_pos)

    joint_acc = ndimage.uniform_filter1d(deepcopy(joint_acc), size=imu_smooth_window, axis=0,  mode="nearest")
    return joint_global_rot, joint_acc


def imu_from_fairmotion(motion):
    # get list of global rotations and global positions 
    joint_idx = [motion.skel.get_index_joint(jn) for jn in imu_constants.imu_joint_names]
    motion_global = motion.to_matrix(local=False)

    return imu_from_global_T(motion_global, joint_idx)

