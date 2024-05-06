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

def _syn_acc_torch(pos, device):

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

# reference: TransPose (SIGGRAPH 2021)
# def _syn_acc_smooth(pos):
#     """
#     Synthesize accelerations from joint positions.
#     """
#     smooth_n = 4
#     mid = smooth_n // 2
#     acc = np.stack([(pos[i] + pos[i + 2] - 2 * pos[i + 1]) * 3600 for i in range(0, pos.shape[0] - 2)]) # Shape: [N-2, # of posi_mask, 3]
#     acc = np.concatenate((np.zeros_like(acc[:1]), acc, np.zeros_like(acc[:1]))) # Shape: [N, # of posi_mask, 3]

#     # this way of smoothing is not mandatory, in TIP it just uses p(t-n) + p(t+n) - 2p(t) / (nDT)^2, as in TransPose paper. can later change to this..
#     if mid != 0:
#         acc[smooth_n:-smooth_n] = np.stack(
#             [(pos[i] + pos[i + smooth_n * 2] - 2 * pos[i + smooth_n]) * 3600 / smooth_n ** 2
#                 for i in range(0, pos.shape[0] - smooth_n * 2)])
#     return acc

# def _syn_acc(v):
#     r"""
#     Synthesize accelerations from joint positions.
#     """
#     smooth_n = 4
#     mid = smooth_n // 2
#     acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)]) # torch.Size([N-2, # of vi_mask, 3])
#     acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1]))) # torch.Size([N, # of vi_mask, 3])

#     # this way of smoothing is not mandatory, in TIP it just uses p(t-n) + p(t+n) - 2p(t) / (nDT)^2, as in TransPose paper. can later change to this..
#     if mid != 0:
#         acc[smooth_n:-smooth_n] = torch.stack(
#             [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
#                 for i in range(0, v.shape[0] - smooth_n * 2)])
#     return acc

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


# simple double integration
def imu_acc_integrate(sensor_acc, fps=30, start_pos=np.zeros(3), start_vel=np.zeros(3)):
    dt = 1 / fps

    # initializing velocity and position arrays
    velocity = np.zeros_like(sensor_acc)
    position = np.zeros_like(sensor_acc)

    position[0] = start_pos
    velocity[0] = start_vel

    # single integration of acceleration to get velocity
    for i in range(1, sensor_acc.shape[0]):
        velocity[i] = velocity[i-1] + sensor_acc[i] * dt

    # single integration of velocity to get position
    for i in range(1, velocity.shape[0]):
        position[i] = position[i-1] + velocity[i] * dt

    return position