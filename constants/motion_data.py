"""
Parsing AMASS data related
"""

FPS = 30.0
BETA_DIM = 10
BM_PATH = "../data/smpl_models/smplx/SMPLX_NEUTRAL.npz" # smplx
SMPLH_BM_PATH = "../data/smpl_models/smplh/male/model.npz"
# Custom names for 22 joints in AMASS data

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

JOINT_NAMES = [
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

EE_JOINTS = [
    "LeftHand",
    "RightHand",
    "LeftFoot",
    "RightFoot",
    "Head",
]

FOOT_JOINTS = [
    # "LeftHand",
    # "RightHand",
    "LeftFoot",
    "RightFoot",
    # "Head"
]

# add to solve overfitting on foot
LEG_JOINTS = [
    "LeftLeg",
    "RightLeg", 
]

HAND_JOINTS = [
    "LeftHand",
    "RightHand",
    # "LeftFoot",
    # "RightFoot",
    # "Head"
]

NUM_JOINTS = len(JOINT_NAMES)

UP_AXIS = "z"
height_indice = 2 if UP_AXIS == "z" else 1
import numpy as np
plane = np.array([1,0,1]) if UP_AXIS == "y" else np.array([1,1,0])
facing_dir_env = np.array([0,0,1]) if UP_AXIS == "y" else np.array([0,-1,0])

# preprocess related
preprocess_window = 40
preprocess_offset = 3

# smpl vertices
smpl_forehead_vertices = [
    2088,
    2091,
    2176,
    2177,
    2178,
    2179,
    8939,
    8968,
    8984,
    9184,
    9187,
    9238,
    9328,
    9330,
]

# constants related to realdata/system setup

# head2cam 
head2cam = np.eye(4)
head2cam[:3,3] = np.array([-2.75976617e-04, 9.94532176e-02, 1.33585830e-01])

cam2head = np.array(
[[-0.99566608, -0.03612037,  0.0856993,   0.01008574],
 [ 0.01588535, -0.97400612, -0.22596402,  0.19671741], 
 [0.09163355, -0.22362335,  0.97035864, -0.09213118],
 [ 0.,          0.,          0.,          1.,        ]]
)

head_ori_tpose = np.array(
[[-0.9995738 ,  0.0185932 , -0.02250584],
    [-0.02185223,  0.03463845,  0.99916098],
    [ 0.01935716,  0.99922694, -0.03421738]]
)

# head_ori_tpose = np.array(
#     [[-1.0, 0.0, 0.0],
#      [0.0, 0.0, 1.0],
#      [0.0, 1.0, 0.0]]
# )

joint_ori_tpose = np.array(
    [[-1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0]]
)


# constants related to contact
contact_joints = [
    "Hips",
    "LeftFoot",
    "RightFoot"
]

contact_joint_idx = [JOINT_NAMES.index(cjoint) for cjoint in contact_joints]

contact_vel = 0.04 

contact_pivot = 0

foot_joint_idx = [JOINT_NAMES.index(fjoint) for fjoint in FOOT_JOINTS]

imu_hand_joint_idx = [JOINT_NAMES.index(hjoint) for hjoint in HAND_JOINTS]

toe_joints = [
    "LeftToe",
    "RightToe"
]

toe_joints_idx = [JOINT_NAMES.index(tjoint) for tjoint in toe_joints]

contact_vel_threshold = 0.008

head_neck_joint_idx = [JOINT_NAMES.index("Head"), JOINT_NAMES.index("Neck")]

head_joint_idx = JOINT_NAMES.index("Head")
# path
# REALDATA_PATH = "/data/realdata"
# BASE_PATH = "/home/jiye/Desktop/TotalCapture"


# for rendering smplx mesh
# default hand pose
import torch
pose_hand = torch.Tensor([ 0.1117,  0.0429, -0.4164,  0.1088, -0.0660, -0.7562, -0.0964, -0.0909,
        -0.1885, -0.1181,  0.0509, -0.5296, -0.1437,  0.0552, -0.7049, -0.0192,
        -0.0923, -0.3379, -0.4570, -0.1963, -0.6255, -0.2147, -0.0660, -0.5069,
        -0.3697, -0.0603, -0.0795, -0.1419, -0.0859, -0.6355, -0.3033, -0.0579,
        -0.6314, -0.1761, -0.1321, -0.3734,  0.8510,  0.2769, -0.0915, -0.4998,
         0.0266,  0.0529,  0.5356,  0.0460, -0.2774,  0.1117, -0.0429,  0.4164,
         0.1088,  0.0660,  0.7562, -0.0964,  0.0909,  0.1885, -0.1181, -0.0509,
         0.5296, -0.1437, -0.0552,  0.7049, -0.0192,  0.0923,  0.3379, -0.4570,
         0.1963,  0.6255, -0.2147,  0.0660,  0.5069, -0.3697,  0.0603,  0.0795,
        -0.1419,  0.0859,  0.6355, -0.3033,  0.0579,  0.6314, -0.1761,  0.1321,
         0.3734,  0.8510, -0.2769,  0.0915, -0.4998, -0.0266, -0.0529,  0.5356,
        -0.0460,  0.2774])
