# partially referred to https://github.com/jyf588/transformer-inertial-poser/blob/master/preprocess_DIP_TC_new.py
from copy import deepcopy
from matplotlib.pyplot import axes
import torch
import sys, os
# sys.path.insert(0, os.path.dirname(__file__))
# sys.path.append("..")
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

import torch
import pickle 
import numpy as np
from IPython import embed 
import constants.motion_data as motion_constants
import constants.imu as imu_constants 
from fairmotion.utils import utils
import imu2body_eval.imu as imu 
# from realdata.utils import *
from scipy import ndimage 
from fairmotion.ops import conversions 
from constants.path import *


import imu2body_eval.amass_smplh as amass
body_model = amass.load_body_model(bm_path=motion_constants.SMPLH_BM_PATH)
skel_with_offset = amass.create_skeleton_from_amass_bodymodel(bm=body_model)	
skel = skel_with_offset[0]


imu_joint_names = imu_constants.imu_joint_names
imu_joint_idx = [skel.get_index_joint(jn) for jn in imu_joint_names]

# vis
result_dict = {}
result_dict['motions'] = []
result_dict['imu_ori'] = []
result_dict['imu_acc'] = []

def get_imu_from_tc(tc_file):

    tc_file = os.path.join(TC_DIR, tc_file)
    with open(tc_file, "rb") as f:
        data = pickle.load(f,  encoding="latin1")

    # 30fps 
    stride = 2
    data['ori'] = data['ori'][::stride]
    data['acc'] = data['acc'][::stride]

    rot_tc_up_env = conversions.A2R(np.array([np.pi / 2, 0, 0]))
    real_imu_acc = data['acc'][:,[0,1],:] # (lw, rw, ll, rl, h, r)
    real_imu_ori = data['ori'][:,[0,1],:] # (lw, rw, ll, rl, h, r)
    
    real_imu_ori = np.einsum('jk,abki->abji', rot_tc_up_env, real_imu_ori)
    real_imu_acc = np.einsum('jk,abk->abj', rot_tc_up_env, real_imu_acc)

    return real_imu_ori, real_imu_acc
