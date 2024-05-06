import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
from fairmotion.core import motion as motion_class
from fairmotion.ops import conversions, motion as motion_ops
from fairmotion.utils import utils
import numpy as np
from copy import deepcopy 
from IPython import embed 
import math
from tqdm import tqdm

foot_joints = ["UpLeg", "Leg", "Foot"]

joint_idx = {
    "LeftFoot": 7,
    "RightFoot": 8,
}

class FootCleanup():
    def __init__(self, motion, contact_label) -> None:
        self.motion = motion 
        self.motion_global_T = self.motion.to_matrix(local=False)
        self.contact_label = {
            "Left": contact_label[:,0],
            "Right": contact_label[:,1]
        }

        # load foot joints
        self.foot_joints = {}
        self.dir = ["Left", "Right"]
        for dir in self.dir:
            dir_foot_joints = []
            for fj in foot_joints:
                dir_foot_joints.append(f"{dir}{fj}")
            self.foot_joints[dir] = dir_foot_joints
        self.toe_joints = ["LeftToe", "RightToe"]

        # thresholds 
        self.foot_vel_threshold = 0.04

        # blend related
        self.blend_frames = 5

    def get_foot_pos_in_motion(self, joint_idx, frame_idx):
        return self.motion_global_T[frame_idx,joint_idx,:3,3]

    def generate_foot_target(self, start, end):
        targets = []
        stop_add_flag = False 

        for i in range(start, end):
            if self.cur_foot_vel_norm[i] > self.foot_vel_threshold:
                stop_add_flag = True 
            if not stop_add_flag:
                targets.append(self.cur_foot_pos[i])
        
        targets = np.array(targets)
        targets = np.mean(targets, axis=0)
        return targets 
    
    def foot_cleanup(self):
        
        self.contact_pairs = {}

        for dir in self.dir:
            foot_idx = self.motion.skel.get_index_joint(f"{dir}Foot")
            self.cur_foot_pos = self.motion_global_T[:,foot_idx,:3,3]
            self.cur_foot_vel = self.motion_global_T[1:,foot_idx,:3,3] - self.motion_global_T[:-1,foot_idx,:3,3]
            self.cur_foot_vel_norm = np.linalg.norm(self.cur_foot_vel, axis=-1)

            self.contact_pairs[dir] = []
            contact_label_by_dir = self.contact_label[dir]
            contact_pair = {}
            frame_start = 0
            for frame_idx in tqdm(range(1, self.motion.num_frames())):
                prev_contact = contact_label_by_dir[frame_idx-1]
                cur_contact = contact_label_by_dir[frame_idx]
                if prev_contact and not cur_contact:
                    start = frame_start
                    end = frame_idx 
                    if (end - start) < self.blend_frames*2:
                        continue 
                    # generate target
                    target = self.generate_foot_target(start, end)
                    try:
                        self.glue_foot(self.motion, target_foot_pos=target, start=start, end=end, dir=dir)
                    except:
                        print(f"skip glue-foot in frame:{start} and {end}")
                        continue
                    contact_pair['start'] = start 
                    contact_pair['end'] = end 
                    contact_pair['target'] = target
                    self.contact_pairs[dir].append(deepcopy(contact_pair))

                elif not prev_contact and cur_contact:
                    frame_start = frame_idx




    # previous version
    def glue_foot(self, motion, target_foot_pos, start, end, dir):
        blend_frames = self.blend_frames
        smooth_before_duration = min(start, blend_frames)
        smooth_end_duration = min(len(motion.poses)-end-1, blend_frames)

        smooth_flag = False
        for i in range(start, end):
            # self.two_joint_ik(motion, i, target_foot_pos, dir)

            # smooth before
            if i >= start and i < start + smooth_before_duration:
                idx = i - start-1
                weight = idx / float(smooth_before_duration)
                weight = 0.5 * np.cos(math.pi * weight) + 0.5
                cur_pos = conversions.T2p(motion.poses[i].get_transform(f"{dir}Foot", local=False))
                target_foot_pos = (weight) * cur_pos + (1-weight)* target_foot_pos
                smooth_flag = True

            # smooth after 
            if i > (end-smooth_end_duration) and i <= end:
                idx = end - i
                weight = idx / float(smooth_end_duration)
                weight = 0.5 * np.cos(math.pi * weight) + 0.5
                cur_pos = conversions.T2p(motion.poses[i].get_transform(f"{dir}Foot", local=False))
                target_foot_pos = weight * cur_pos + (1-weight) * target_foot_pos
                smooth_flag = True

            # if smooth_flag:
            self.two_joint_ik(motion, i, target_foot_pos, dir)
            
            smooth_flag = False
    
    # originally from https://theorangeduck.com/page/simple-two-joint
    def two_joint_ik(self, motion, frame, target_foot_pos, dir):

        a_idx = motion.skel.get_index_joint(f"{dir}UpLeg")
        b_idx = motion.skel.get_index_joint(f"{dir}Leg")
        c_idx = motion.skel.get_index_joint(f"{dir}Foot")

        a_iso = motion.poses[frame].get_transform(key=f"{dir}UpLeg", local=False)
        b_iso = motion.poses[frame].get_transform(key=f"{dir}Leg", local=False)
        c_iso = motion.poses[frame].get_transform(key=f"{dir}Foot", local=False)

        a_pos = conversions.T2p(a_iso)
        b_pos = conversions.T2p(b_iso)
        c_pos = conversions.T2p(c_iso)

        lab = np.linalg.norm(b_pos - a_pos)
        lcb = np.linalg.norm(b_pos - c_pos)
        lat = np.clip(np.linalg.norm(target_foot_pos - a_pos), 0.01, lab+lcb-0.01)

        c_a = (c_pos - a_pos) / np.linalg.norm(c_pos - a_pos)
        b_a = (b_pos - a_pos) / np.linalg.norm(b_pos - a_pos)
        a_b = (a_pos - b_pos) / np.linalg.norm(a_pos - b_pos)
        c_b = (c_pos - b_pos) / np.linalg.norm(c_pos - b_pos)
        t_a = (target_foot_pos - a_pos) / np.linalg.norm(target_foot_pos - a_pos)

        ac_ab_0 = np.arccos(np.clip(np.dot(c_a, b_a), -1.0, 1.0))
        ba_bc_0 = np.arccos(np.clip(np.dot(a_b, c_b), -1.0, 1.0))
        ac_at_0 = np.arccos(np.clip(np.dot(c_a, t_a), -1.0, 1.0))

        ac_ab_1 = np.arccos(np.clip((lcb*lcb-lab*lab-lat*lat) / (-2*lab*lat), -1.0, 1.0))
        ba_bc_1 = np.arccos(np.clip((lat*lat-lab*lab-lcb*lcb) / (-2*lab*lcb), -1.0, 1.0))

        axis0 = np.cross(c_a, b_a) 
        axis0 /= (np.linalg.norm(axis0) + 1e-05)

        axis1 = np.cross(c_a, t_a)
        axis1 /= (np.linalg.norm(axis1) + 1e-05)

        # r0_axis = np.dot(np.linalg.inv(a_iso[:3, :3]), axis0)
        # r1_axis = np.dot(np.linalg.inv(b_iso[:3, :3]), axis0)
        # r2_axis = np.dot(np.linalg.inv(a_iso[:3, :3]), axis1)

        r0_axis = np.dot(a_iso[:3, :3].swapaxes(-2,-1), axis0)
        r1_axis = np.dot(b_iso[:3, :3].swapaxes(-2,-1), axis0)
        r2_axis = np.dot(a_iso[:3, :3].swapaxes(-2,-1), axis1)

        r0_angle = ac_ab_1 - ac_ab_0
        r1_angle = ba_bc_1 - ba_bc_0
        r2_angle = ac_at_0

        # get original pos and replace 
        original_mat = motion_class.Pose.to_matrix(motion.poses[frame])

        original_mat[a_idx, :3, :3] = original_mat[a_idx,:3,:3] @ conversions.A2R(r0_axis*r0_angle) @ conversions.A2R(r2_axis*r2_angle)
        original_mat[b_idx, :3, :3] = original_mat[b_idx,:3,:3] @ conversions.A2R(r1_axis*r1_angle)

        fixed_pose = motion_class.Pose.from_matrix(original_mat, deepcopy(motion.skel))

        motion.poses[frame] = fixed_pose
