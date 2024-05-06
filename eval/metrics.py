# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Metric functions with same inputs
# this is from the AGRoL (CVPR 2023) repo

import numpy as np
import torch
from IPython import embed

# hand_idx = [20, 21]
def pred_jitter(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    hand_idx,
    foot_idx,
    fps,
    root_rel=False
):
    # joint_idx = upper_index + lower_index 
    # joint_idx.sort()
    # predicted_position  = predicted_position[:,joint_idx,:]

    pred_jitter = (
        (
            (
                predicted_position[3:]
                - 3 * predicted_position[2:-1]
                + 3 * predicted_position[1:-2]
                - predicted_position[:-3]
            )
            * (fps**3)
        )
        .norm(dim=2)
        .mean()
    )
    return pred_jitter


def gt_jitter(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    hand_idx,
    foot_idx,
    fps,
    root_rel=False
):
    # joint_idx = upper_index + lower_index 
    # joint_idx.sort()
    # gt_position  = gt_position[:,joint_idx,:]

    # [batch-3, seq, 22, 3]
    gt_jitter = (
        (
            (
                gt_position[3:]
                - 3 * gt_position[2:-1]
                + 3 * gt_position[1:-2]
                - gt_position[:-3]
            )
            * (fps**3)
        )
        .norm(dim=2)
        .mean()
    )
    return gt_jitter


def mpjre(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    hand_idx,
    foot_idx,
    fps,
    root_rel=False
):
    diff = gt_angle - predicted_angle
    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    rot_error = torch.mean(torch.absolute(diff))
    return rot_error


def mpjpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    hand_idx,
    foot_idx,
    fps,
    root_rel=False
):
    joint_idx = upper_index + lower_index
    pos_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position[:,joint_idx,:] - predicted_position[:,joint_idx,:]), axis=-1))
    )
    return pos_error


def root_mpjpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    hand_idx,
    foot_idx,
    fps,
    root_rel=False
):
    joint_idx = upper_index + lower_index
    joint_idx.sort()
    gt_position = gt_position[:,joint_idx,:]
    predicted_position = predicted_position[:,joint_idx,:]
    root_relative_gt_pos = gt_position[:,1:,...] - gt_position[:,0:1,...]
    root_relative_pred_pos = predicted_position[:,1:,...] - predicted_position[:,0:1,...]
    pos_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(root_relative_gt_pos - root_relative_pred_pos), axis=-1))
    )
    return pos_error


def handpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    hand_idx,
    foot_idx,
    fps,
    root_rel=False
):
    if root_rel:
        root_relative_gt_pos = gt_position- gt_position[:,0:1,...]
        root_relative_pred_pos = predicted_position - predicted_position[:,0:1,...]

        pos_error_hands = torch.mean(
            torch.sqrt(torch.sum(torch.square(root_relative_gt_pos - root_relative_pred_pos), axis=-1))[
                ..., hand_idx
            ]
        )
        return pos_error_hands

    pos_error_hands = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., hand_idx
        ]
    )
    return pos_error_hands


def upperpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    hand_idx,
    foot_idx,
    fps,
    root_rel=False
):
    if root_rel:
        root_relative_gt_pos = gt_position- gt_position[:,0:1,...]
        root_relative_pred_pos = predicted_position - predicted_position[:,0:1,...]
        upper_body_error = torch.mean(
            torch.sqrt(torch.sum(torch.square(root_relative_gt_pos - root_relative_pred_pos), axis=-1))[
                ..., upper_index
            ]
        )
        return upper_body_error

    upper_body_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., upper_index
        ]
    )
    return upper_body_error


def lowerpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    hand_idx,
    foot_idx,
    fps,
    root_rel=False
):
    if root_rel:
        root_relative_gt_pos = gt_position- gt_position[:,0:1,...]
        root_relative_pred_pos = predicted_position - predicted_position[:,0:1,...]

        lower_body_error = torch.mean(
            torch.sqrt(torch.sum(torch.square(root_relative_gt_pos - root_relative_pred_pos), axis=-1))[
                ..., lower_index
            ]
        )
        return lower_body_error


    lower_body_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., lower_index
        ]
    )
    return lower_body_error


def rootpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    hand_idx,
    foot_idx,
    fps,
    root_rel=False
):
    pos_error_root = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., [0]
        ]
    )
    return pos_error_root


def mpjve(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    hand_idx,
    foot_idx,
    fps,
    root_rel=False
):
    joint_idx = upper_index + lower_index
    gt_velocity = (gt_position[1:, ...] - gt_position[:-1, ...]) * fps
    predicted_velocity = (
        predicted_position[1:, ...] - predicted_position[:-1, ...]
    ) * fps
    vel_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_velocity[:,joint_idx,:] - predicted_velocity[:,joint_idx,:]), axis=-1))
    )
    return vel_error


# contact (foot slip) metric 
def contact(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    hand_idx,
    foot_idx,
    fps,
    root_rel=False
):
    # foot_idx = [7, 8]
    
    foot_slip_avg = 0.0
    for idx, foot in enumerate(foot_idx):
        foot_pos = predicted_position[:,foot,:]

        # foot_contact_mask = foot_pos[...,2] < 0.06
        foot_vel = torch.norm(predicted_position[1:,foot,:] - predicted_position[:-1,foot,:], dim=-1)

        foot_vel = torch.cat((foot_vel[0:1], foot_vel), dim=0)
        # get opposite foot
        opp_foot_idx = (idx+1) % 2
        opp_foot_pos = predicted_position[:,opp_foot_idx,:]
        opp_foot_vel = torch.norm(opp_foot_pos[1:] - opp_foot_pos[:-1], dim=-1)

        foot_contact_mask = foot_vel < 0.02 
        for i in range(foot_contact_mask.shape[0]):
            if not foot_contact_mask[i]:
                if foot_vel[i] > 0.05 and opp_foot_vel[i] > 0.05:
                    continue
                if foot_vel[i] < opp_foot_vel[i]:
                    foot_contact_mask[i] = True 
        # embed()
        foot_vel_on_contact = foot_vel * foot_contact_mask

        foot_slip = torch.mean(foot_vel_on_contact)
        foot_slip_avg += foot_slip
    
    return foot_slip_avg * 0.5 * fps
    

metric_funcs_dict = {
    "mpjre": mpjre,
    "mpjpe": mpjpe,
    "mpjve": mpjve,
    "handpe": handpe,
    "upperpe": upperpe,
    "lowerpe": lowerpe,
    "rootpe": rootpe,
    "pred_jitter": pred_jitter,
    "gt_jitter": gt_jitter,
    "contact": contact,
    "root_mpjpe": root_mpjpe
}

import math
RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0

pred_metrics = [
    "mpjre",
    "mpjpe",
    "mpjve",
    "handpe",
    "upperpe",
    "lowerpe",
    "rootpe",
    "pred_jitter",
    "contact"
]
gt_metrics = [
    "gt_jitter",
]
all_metrics = pred_metrics + gt_metrics

metrics_coeffs = {
    "contact": METERS_TO_CENTIMETERS,
    "mpjre": RADIANS_TO_DEGREES,
    "mpjpe": METERS_TO_CENTIMETERS,
    "mpjve": METERS_TO_CENTIMETERS,
    "handpe": METERS_TO_CENTIMETERS,
    "upperpe": METERS_TO_CENTIMETERS,
    "lowerpe": METERS_TO_CENTIMETERS,
    "rootpe": METERS_TO_CENTIMETERS,
    "pred_jitter": 1.0,
    "gt_jitter": 1.0,
    "gt_mpjpe": METERS_TO_CENTIMETERS,
    "gt_mpjve": METERS_TO_CENTIMETERS,
    "gt_handpe": METERS_TO_CENTIMETERS,
    "gt_rootpe": METERS_TO_CENTIMETERS,
    "gt_upperpe": METERS_TO_CENTIMETERS,
    "gt_lowerpe": METERS_TO_CENTIMETERS,
    "root_mpjpe": METERS_TO_CENTIMETERS,
}


def get_metric_function(metric):
    return metric_funcs_dict[metric]
