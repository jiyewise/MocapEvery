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
import open3d as o3d
import argparse
import json


import imu2body_eval.amass_smplh as amass
body_model = amass.load_body_model(bm_path=motion_constants.SMPLH_BM_PATH)
skel_with_offset = amass.create_skeleton_from_amass_bodymodel(bm=body_model)	
skel = skel_with_offset[0]


def load_motion_and_scene(filename, only_read_motion=True):

    file = filename.split("/")[-1]
    file = file.replace(".pkl", "")

    smpl_params = pickle.load(open(filename, "rb"))
    bdata = {}
    bdata['mocap_framerate'] = 30.0
    bdata['poses'] = smpl_params['poses']
    bdata['trans'] = smpl_params['transes']
    
    amass_to_fairmotion = amass.create_motion_from_amass_bdata(bdata=bdata, bm=body_model, skel_with_offset=deepcopy(skel_with_offset))

    if only_read_motion:
        return amass_to_fairmotion

