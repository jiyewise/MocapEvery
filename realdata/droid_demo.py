import sys,os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('../libraries/DROID-SLAM/droid_slam')
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F
from IPython import embed 

from fairmotion.ops import conversions
from realdata.img_utils import *
from visualization import *
from fairmotion.utils import utils as futils
import constants.path as path_constants

def image_stream(imagedir, calib, stride, write_unproj_dir=""):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])
        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        # saved resized & unprojected image
        if write_unproj_dir != "":
            img_to_save = deepcopy(image).permute(1, 2, 0).cpu().numpy()
            t_str = str(t).zfill(5)
            cv2.imwrite(f"{write_unproj_dir}/undistort_{t_str}.png", img_to_save.astype(np.uint8))

        yield t, image[None], intrinsics


# def save_reconstruction(traj_est, droid, reconstruction_path):
#     # traj_est = traj_est.cpu().numpy()

#     t = droid.video.counter.value

#     embed()

#     tstamps = droid.video.tstamp[:t].cpu().numpy()
#     images = droid.video.images[:t].cpu().numpy()
#     disps = droid.video.disps_up[:t].cpu().numpy()
#     poses = droid.video.poses[:t].cpu().numpy()
#     intrinsics = droid.video.intrinsics[:t].cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", type=str, help="data name")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")

    from realdata.droid_args import add_droid_argparse
    parser = add_droid_argparse(parser=parser)

    args = parser.parse_args()
    args.stereo = False

    image_dir = os.path.join(path_constants.DATA_PATH, f"{args.data_name}/sfm/images")
    calib = os.path.join(path_constants.DATA_PATH, f"{args.data_name}/sfm/calib.txt")
    reproject_path = os.path.join(path_constants.DATA_PATH, f"{args.data_name}/sfm/reprojected/")
    pc_path = os.path.join(path_constants.DATA_PATH, f"{args.data_name}/sfm/pointcloud/")
    undistort_img_path = os.path.join(path_constants.DATA_PATH, f"{args.data_name}/sfm/undistorted_images")
    undistort_calib_path = os.path.join(path_constants.DATA_PATH, f"{args.data_name}/sfm/undistort_calib.txt")

    futils.create_dir_if_absent(reproject_path)
    futils.create_dir_if_absent(pc_path)
    futils.create_dir_if_absent(undistort_img_path)

    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    args.upsample = True

    # intrinsics (assume that image size/intrinsics are fixed throughout the whole video)
    intrinsics_save = None

    tstamps = []
    for (t, image, intrinsics) in tqdm(image_stream(image_dir, calib, args.stride, undistort_img_path)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        # save edited intrinsics for undistorted images
        if intrinsics_save is None:
            intrinsics_np = intrinsics.clone().detach().cpu().numpy()
            intrinsics_save = intrinsics_np

        droid.track(t, image, intrinsics=intrinsics)


    traj_est = droid.terminate(image_stream(image_dir, calib, args.stride))
    pcd_data = droid.get_pointcloud_data()

    # save camera intrinsics 
    with open(undistort_calib_path, 'w') as f:
        f.write(' '.join(map(str, intrinsics_np.tolist())))    

    # save pointcloud 
    pts_result, clr_result = pcd_data
    result_point_actor = create_point_actor(pts_result, clr_result)
    import open3d as o3d
    o3d.io.write_point_cloud(os.path.join(pc_path, "pcd_results.ply"), result_point_actor)
    print("finished saving pointcloud")
    
    # save camera trajectory
    cam_traj_path = os.path.join(path_constants.DATA_PATH, f"{args.data_name}/cam_traj.pkl")
    pickle.dump(traj_est, open(cam_traj_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print("finished saving camera trajectory")
