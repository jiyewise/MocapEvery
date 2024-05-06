import torch
import numpy as np
import torch.nn as nn
from IPython import embed
import math
from fairmotion.ops import conversions
from fairmotion.utils import constants
from pytorch3d import transforms
from fairmotion.data import bvh
from fairmotion.core import motion as motion_classes
from fairmotion.utils import utils as fairmotion_utils
from copy import deepcopy
import os

# this code is from pytorch3d
def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(transforms.matrix_to_quaternion(matrix))

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def vertice_transform(vertice, transform):
    assert vertice.shape[0] == transform.shape[0], "check vertice batch and transform batch shape"
    verts_batch_homo = nn.functional.pad(vertice, (0,1), mode='constant', value=1)
    verts_batch_homo_transformed = torch.matmul(verts_batch_homo,
                                                transform.permute(0,2,1))

    verts_batch_transformed = verts_batch_homo_transformed[:,:,:-1]
    
    return verts_batch_transformed


def denormalize(seq, mean, std, device):
    # all the input should be in tensor form, save device
    seq_dim = seq.shape[-1]
    epsilon = torch.full((seq_dim,), constants.EPSILON).to(device)
    denorm_seq = seq * (epsilon + std.to(device)) + mean.to(device)
    return denorm_seq

def get_linear_teacher_forcing_ratio(epoch, total_epoch, scale=20):
    return (1 - scale*epoch / total_epoch)

def get_exp_teacher_forcing_ratio(epoch, scale=0.1, lower_limit=0.05):
    ratio = np.exp(-1*scale*epoch)
    return 0 if (ratio) < lower_limit else ratio

def divide_and_fk(seq, root_start_idx, rot_start_idx, rep, skel, parent, zero_root=None):
    # should be denormalized first
    batch, seq_len, _ = seq.shape
    denorm_rotations_pred = seq[...,rot_start_idx:]
    denorm_root_pos_pred =  zero_root if zero_root is not None else seq[...,root_start_idx:rot_start_idx]
    

    if rep == "6d":
        # convert output
        denorm_rotations_pred_ = denorm_rotations_pred.reshape(batch, seq_len, -1, 6) # map to 6d
        denorm_rotations_pred_ = transforms.rotation_6d_to_matrix(denorm_rotations_pred_)
        output_pos = rot_matrix_fk_tensor(denorm_rotations_pred_, denorm_root_pos_pred, skel, parent)

    elif rep == "quat":
        # convert output
        denorm_rotations_pred_ = denorm_rotations_pred.reshape(batch, seq_len, -1, 4) # map to quaternions
        denorm_rotations_pred_ = denorm_rotations_pred_ / torch.norm(denorm_rotations_pred_, dim=-1, keepdim=True) # normalize
        output_pos = quat_fk_tensor(denorm_rotations_pred_, denorm_root_pos_pred, skel, parent)
    else:
        assert False, "representation should be either 6d or quat"

    # test rot tensor fk
    # denorm_matrix = transforms.rotation_6d_to_matrix(denorm_rotations_pred.reshape(batch, seq_len, -1, 6))
    # gp = rot_matrix_fk_tensor(denorm_matrix, denorm_root_pos_pred, skel, parent, denorm_rotations_pred_)
    # gp_quat = quat_fk_tensor(denorm_rotations_pred_, denorm_root_pos_pred, skel, parent)

    root_info = seq[..., root_start_idx:rot_start_idx] if zero_root is not None else denorm_root_pos_pred
    return output_pos, root_info, denorm_rotations_pred_


# move to utils
def write_result(render_data, skel, base_directory, subdirectory, idx=0):
    length = len(render_data)
    motions = []
    # m = bvh.load(self.config['skel']['bvh'], scale=0.01)
    m = bvh.load(skel, scale=1)
    skel = deepcopy(m.skel)
    for i in range(length):

        assert len(render_data) > 0, "Render data is empty"
        output = render_data[i].output_T
        gt = render_data[i].gt_T

        motion_output = motion_classes.Motion.from_matrix(output, skel)
        motion_gt = motion_classes.Motion.from_matrix(gt, skel)
        motion_output.set_fps(30)
        motion_gt.set_fps(30)

        # save 
        # if self.is_optimize:
        # 	subdirectory = "optimize_contact2/"
        # elif self.is_custom:
        # 	subdirectory = self.bvh_filepath.strip().split("/")[-1][:-4]
        # else:
        # 	subdirectory = "testset/"

        gt_path = os.path.join(base_directory, subdirectory) + f"/ref_{idx}/"
        output_path = os.path.join(base_directory, subdirectory) + f"/pred_{idx}/"
        fairmotion_utils.create_dir_if_absent(os.path.dirname(gt_path))
        fairmotion_utils.create_dir_if_absent(os.path.dirname(output_path))

        bvh.save(
            motion_output,
            filename=output_path+"output_{}.bvh".format(i)
        )
        
        bvh.save(
            motion_gt,
            filename=gt_path+"gt_{}.bvh".format(i)
        )

        pair = [motion_output, motion_gt]
        motions.append(pair)

def length(x, axis=-1, keepdims=True):
    """
    Computes vector norm along a tensor axis(axes)
    :param x: tensor
    :param axis: axis(axes) along which to compute the norm
    :param keepdims: indicates if the dimension(s) on axis should be kept
    :return: The length or vector of lengths.
    """
    lgth = np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims))
    return lgth


def normalize(x, axis=-1, eps=1e-8):
    """
    Normalizes a tensor over some axis (axes)
    :param x: data tensor
    :param axis: axis(axes) along which to compute the norm
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized tensor
    """
    res = x / (length(x, axis=axis) + eps)
    return res


def quat_normalize(x, eps=1e-8):
    """
    Normalizes a quaternion tensor
    :param x: data tensor
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized quaternions tensor
    """
    res = normalize(x, eps=eps)
    return res


def angle_axis_to_quat(angle, axis):
    """
    Converts from an angle-axis representation to a quaternion representation
    :param angle: angles tensor
    :param axis: axis tensor
    :return: quaternion tensor
    """
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q

def quat_to_angle_axis(quat):
    """
    Converts from a quaternion representation to an angle-axis representation
    :param quat: quaternion tensor [T, J, 4]
    :return: angle-axis tensor [T, J, 3]
    """

    axis = quat[..., 1:4]
    norm = np.linalg.norm(axis, axis=-1)
    norm = norm[..., np.newaxis]

    axis = np.where(norm < 1e-06, [0, 0, 0], axis)
    norm = np.where(norm < 1e-06, [1], norm)
    normalized = axis/norm
    
    angle = 2*np.arccos(quat[..., 0])
    angle = np.fmod(angle+np.pi, 2*np.pi)-np.pi
    aa = angle[..., np.newaxis] * normalized
    return aa # [T, J, 3]

def quat_to_T(quat, root_pos):
    """
    This function transforms quat + root pos into transform matrix
    p is set to zero except for the root
    quat.shape = [#, J, 4]
    rootPos.shape = [#, 3] (# should be same as quat)
    return.shape = [#, J, 4, 4]
    """
    p = np.zeros(quat.shape[:-1] + (3,))
    p[:,0,:] = root_pos
    seq_T = conversions.Ap2T(quat_to_angle_axis(quat), p)

    return seq_T

def rotmat_to_T(rot, root):
    seq_len, joint, _, _ = rot.shape
    seq_T = np.tile(np.eye(4)[np.newaxis, np.newaxis], (seq_len, joint, 1, 1))
    seq_T[...,:3,:3] = rot
    seq_T[:,0,:3,3] = root
    return seq_T

def quat_to_logmap(quat, root_pos):
    """
    quat.shape = [#, J, 4]
    rootPos.shape = [#, 3] (# should be same as quat)
    return: aa.shape = [#, 3+4J]
    """
    # n = np.linalg.norm(quat, axis=-1)
    # quat = quat/n[...,np.newaxis]
    aa = quat_to_angle_axis(quat)
    logMap = aa.reshape(aa.shape[0], -1)
    logMap = np.concatenate((root_pos*0.01, logMap), axis=-1, dtype=np.float32)
    return logMap

def logmap_to_quat_and_root(logmap):
    """
    input logmap shape: (#F,3+3J)
    """
    root_pos = logmap[:,:3]*100
    angle_axis = logmap[:,3:]
    angle_axis = angle_axis.reshape(logmap.shape[0], -1, 3)
    rot = conversions.A2R(angle_axis)
    return rot.astype(np.float64), root_pos.astype(np.float64)


def euler_to_quat(e, order='zyx'):
    """
    Converts from an euler representation to a quaternion representation
    :param e: euler tensor
    :param order: order of euler rotations
    :return: quaternion tensor
    """
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)
    }

    q0 = angle_axis_to_quat(e[..., 0], axis[order[0]])
    q1 = angle_axis_to_quat(e[..., 1], axis[order[1]])
    q2 = angle_axis_to_quat(e[..., 2], axis[order[2]])

    return quat_mul(q0, quat_mul(q1, q2))


def quat_inv(q):
    """
    Inverts a tensor of quaternions
    :param q: quaternion tensor
    :return: tensor of inverted quaternions
    """
    res = np.asarray([1, -1, -1, -1], dtype=np.float32) * q
    return res


# PyTorch-backed implementations

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    
    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.reshape(-1, 4, 1), q.reshape(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).reshape(original_shape)

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    if q.shape[-1] != 4:
        print("check q shape in qrot")
        embed()
    if v.shape[-1] != 3:
        print("check v shape in qrot")
        embed()
    if q.shape[:-1] != v.shape[:-1]:
        print("compare q and v shape in qrot")
        embed()
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    original_shape = list(v.shape)
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).reshape(original_shape)


def quat_fk_tensor(lrot, root_pos, offset, parents):
    """
    Perform forward kinematics using the given trajectory and local rotations.
    Arguments (where N = batch size, L = sequence length, J = number of joints):
        -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
        -- root_positions: (N, L, 3) tensor describing the root joint positions.
    """
    gp, gr = [root_pos], [lrot[:,:,0]]
    for i in range(1, len(parents)):
        gp.append(qrot(gr[parents[i]], offset[:,:,i,:]) + gp[parents[i]])
        gr.append(qmul(gr[parents[i]], lrot[:,:,i]))
    
    return torch.stack(gp, dim=3).permute(0, 1, 3, 2)

def rot_matrix_fk_tensor(lrot, root_pos, offset, parents):
    gp, gr = [root_pos], [lrot[:,:,0]]
    # gp_quat, gr_quat = [root_pos], [quat_rot[:,:,0]]
    for i in range(1, len(parents)):
        gp.append((gr[parents[i]] @ offset[:,:,i,:].unsqueeze(-1)).squeeze(-1) + gp[parents[i]])
        gr.append(gr[parents[i]] @ lrot[:,:,i])
        # gp_quat.append(qrot(gr_quat[parents[i]], offset[:,:,i,:]) + gp_quat[parents[i]])
        # gr_quat.append(qmul(gr_quat[parents[i]], quat_rot[:,:,i]))

    return torch.stack(gp, dim=3).permute(0, 1, 3, 2)

# def quat_id_tensor(shape):

def angle_axis_to_quat_tensor(angle, axis):
    c = torch.cos(angle/2).unsqueeze(-1)
    s = torch.sin(angle/2).unsqueeze(-1)
    q = torch.concat((c, s * axis), dim=-1)
    return q

def qinv(q, device):
    inv = torch.Tensor([1, -1, -1, -1]).to(device).double() * q
    return inv

def quat_fk_multiple_skel(lrot, root_pos, offset, parents):

    """
    Performs Forward Kinematics (FK) on local quaternions and local positions to retrieve global representations
    :param lrot: tensor of local quaternions with shape (..., Nb of joints, 4)
    :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of global quaternion, global positions
    """
    offsetTensor = np.repeat(offset, lrot.shape[1], axis=1)

    gp, gr = [root_pos[..., 0:1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], offsetTensor[..., i:i+1, :]) + gp[parents[i]])
        gr.append(quat_mul    (gr[parents[i]], lrot[..., i:i+1, :]))

    res = np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)
    return res

def quat_fk(lrot, root_pos, offset, parents):

    """
    Performs Forward Kinematics (FK) on local quaternions and local positions to retrieve global representations
    :param lrot: tensor of local quaternions with shape (..., Nb of joints, 4)
    :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of global quaternion, global positions
    """
    shape = list(lrot.shape)
    shape[-2::] = [1,1]
    offsetTensor = np.tile(offset, tuple(shape))

    gp, gr = [root_pos[..., 0:1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], offsetTensor[..., i:i+1, :]) + gp[parents[i]]) # root offset is ignored anyway
        gr.append(quat_mul    (gr[parents[i]], lrot[..., i:i+1, :]))

    res = np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)
    return res


def quat_ik(grot, gpos, parents):
    """
    Performs Inverse Kinematics (IK) on global quaternions and global positions to retrieve local representations
    :param grot: tensor of global quaternions with shape (..., Nb of joints, 4)
    :param gpos: tensor of global positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of local quaternion, local positions
    """
    
    res = [
        gpos[..., :1, :],

        np.concatenate([
            grot[..., :1, :],
            quat_mul(quat_inv(grot[..., parents[1:], :]), grot[..., 1:, :]),
        ], axis=-2)
    ]

    return res


def quat_mul(x, y):
    """
    Performs quaternion multiplication on arrays of quaternions
    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

    return res


def quat_mul_vec(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).
    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    t = 2.0 * np.cross(q[..., 1:], x)
    res = x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)

    return res


def quat_slerp(x, y, a):
    """
    Performs spherical linear interpolation (SLERP) between x and y, with proportion a
    :param x: quaternion tensor
    :param y: quaternion tensor
    :param a: indicator (between 0 and 1) of completion of the interpolation.
    :return: tensor of interpolation results
    """
    len = np.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = np.zeros_like(x[..., 0]) + a
    amount0 = np.zeros(a.shape)
    amount1 = np.zeros(a.shape)

    linear = (1.0 - len) < 0.01
    omegas = np.arccos(len[~linear])
    sinoms = np.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = np.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = np.sin(a[~linear] * omegas) / sinoms
    res = amount0[..., np.newaxis] * x + amount1[..., np.newaxis] * y

    return res


def quat_between(x, y):
    """
    Quaternion rotations between two 3D-vector arrays
    :param x: tensor of 3D vectors
    :param y: tensor of 3D vetcors
    :return: tensor of quaternions
    """
    res = np.concatenate([
        np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., np.newaxis] +
        np.sum(x * y, axis=-1)[..., np.newaxis],
        np.cross(x, y)], axis=-1)
    return res


def interpolate_local(lcl_r_mb, lcl_q_mb, n_past, n_future):
    """
    Performs interpolation between 2 frames of an animation sequence.
    The 2 frames are indirectly specified through n_past and n_future.
    SLERP is performed on the quaternions
    LERP is performed on the root's positions.
    :param lcl_r_mb:  Local/Global root positions (B, T, 1, 3)
    :param lcl_q_mb:  Local quaternions (B, T, J, 4)
    :param n_past:    Number of frames of past context
    :param n_future:  Number of frames of future context
    :return: Interpolated root and quats
    """
    # Extract last past frame and target frame
    start_lcl_r_mb = lcl_r_mb[:, n_past - 1, :, :][:, None, :, :]  # (B, 1, J, 3)
    end_lcl_r_mb = lcl_r_mb[:, -n_future, :, :][:, None, :, :]

    start_lcl_q_mb = lcl_q_mb[:, n_past - 1, :, :]
    end_lcl_q_mb = lcl_q_mb[:, -n_future, :, :]

    # LERP Local Positions:
    n_trans = lcl_r_mb.shape[1] - (n_past + n_future)
    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    offset = end_lcl_r_mb - start_lcl_r_mb

    const_trans    = np.tile(start_lcl_r_mb, [1, n_trans + 2, 1, 1])
    inter_lcl_r_mb = const_trans + (interp_ws)[None, :, None, None] * offset

    # SLERP Local Quats:
    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    inter_lcl_q_mb = np.stack(
        [(quat_normalize(quat_slerp(quat_normalize(start_lcl_q_mb), quat_normalize(end_lcl_q_mb), w))) for w in
         interp_ws], axis=1)

    return inter_lcl_r_mb, inter_lcl_q_mb


def remove_quat_discontinuities(rotations):
    """
    Removing quat discontinuities on the time dimension (removing flips)
    :param rotations: Array of quaternions of shape (T, J, 4)
    :return: The processed array without quaternion inversion.
    """
    rots_inv = -rotations

    for i in range(1, rotations.shape[0]):
        # Compare dot products
        replace_mask = np.sum(rotations[i - 1: i] * rotations[i: i + 1], axis=-1) < np.sum(rotations[i - 1: i] * rots_inv[i: i + 1], axis=-1)
        replace_mask = replace_mask[..., np.newaxis]
        rotations[i] = replace_mask * rots_inv[i] + (1.0 - replace_mask) * rotations[i]

    return rotations


# Orient the data (map the root orientation face to z)
def rotate_at_frame(X, Q, skel):
    """
    Re-orients the animation data according to the last frame of past context.
    :param X: tensor of local positions of shape (Batchsize, Timesteps, Joints, 3)
    :param Q: tensor of local quaternions (Batchsize, Timesteps, Joints, 4)
    :param parents: list of parents' indices
    :param n_past: number of frames in the past context
    :return: The rotated positions X and quaternions Q
    """

    root_start_Q = Q[:,0:1,0:1,:]
    forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] \
                 * quat_mul_vec(root_start_Q, np.array([0, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :]) # facing dir in this skel is z
    forward = normalize(forward) # get root forward (root face) direction
    yrot = quat_normalize(quat_between(forward, np.array([0, 0, 1]))) # send forward to z

    new_X_root = quat_mul_vec(yrot, X)
    new_Q_root = quat_mul(yrot, Q[:,:,0:1,:])
    Q[:,:,0:1,:] = new_Q_root

    # calculate global q and global p
    # shape of Q : [1, 80, 22, 4]
    # shape of new_X_root : [1, 80, 1, 3]
    global_q, global_p = quat_fk(Q, new_X_root, skel.offset, skel.parent)

    return new_X_root, Q, global_q, global_p


def extract_feet_contacts_tensor(pos, lfoot_idx, rfoot_idx, velfactor=0.02):
    """
    Extracts binary tensors of feet contacts
    :param pos: tensor of global positions of shape (Timesteps, Joints, 3)
    :param lfoot_idx: indices list of left foot joints
    :param rfoot_idx: indices list of right foot joints
    :param velfactor: velocity threshold to consider a joint moving or not
    :return: binary tensors of left foot contacts and right foot contacts
    """
    lfoot_xyz = (pos[:, 1:, lfoot_idx, :] - pos[:, :-1, lfoot_idx, :]) ** 2 # [seq-1, 2, 3]
    contacts_l = (torch.sum(lfoot_xyz, axis=-1) < velfactor)

    rfoot_xyz = (pos[:, 1:, rfoot_idx, :] - pos[:, :-1, rfoot_idx, :]) ** 2
    contacts_r = (torch.sum(rfoot_xyz, axis=-1) < velfactor)

    # Duplicate the last frame for shape consistency
    contacts_l = torch.cat((contacts_l, contacts_l[:,-1:]), axis=1)
    contacts_r = torch.cat((contacts_r, contacts_r[:,-1:]), axis=1)

    return torch.cat((contacts_l, contacts_r), axis=-1)    

def extract_feet_contacts(pos, lfoot_idx, rfoot_idx, velfactor=0.02):
    """
    Extracts binary tensors of feet contacts
    :param pos: tensor of global positions of shape (Timesteps, Joints, 3)
    :param lfoot_idx: indices list of left foot joints
    :param rfoot_idx: indices list of right foot joints
    :param velfactor: velocity threshold to consider a joint moving or not
    :return: binary tensors of left foot contacts and right foot contacts
    """
    lfoot_xyz = (pos[1:, lfoot_idx, :] - pos[:-1, lfoot_idx, :]) ** 2 # [seq-1, 2, 3]
    contacts_l = (np.sum(lfoot_xyz, axis=-1) < velfactor)

    rfoot_xyz = (pos[1:, rfoot_idx, :] - pos[:-1, rfoot_idx, :]) ** 2
    contacts_r = (np.sum(rfoot_xyz, axis=-1) < velfactor)

    # Duplicate the last frame for shape consistency
    contacts_l = np.concatenate([contacts_l, contacts_l[-1:]], axis=0)
    contacts_r = np.concatenate([contacts_r, contacts_r[-1:]], axis=0)

    return contacts_l, contacts_r