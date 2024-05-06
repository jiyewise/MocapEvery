import numpy as np
from fairmotion.ops import conversions
from IPython import embed 
from tqdm import tqdm
import cv2
from fairmotion.ops import math as fmath
from imu2body.functions import * 
import cv2
import numpy as np
from scipy.spatial import cKDTree

def show_image(image):
	image = image.permute(1, 2, 0).cpu().numpy()
	cv2.imshow('image', image / 255.0)
	cv2.waitKey(1)


def draw_points_on_image(image, reprojected_points, color=(0,0,255)):
	h, w, c = image.shape
	is_drawn = False
	for p_idx, point in enumerate(reprojected_points):
		if point[0] > w or point[0] < 0:
				continue
		if point[1] > h or point[1] < 0:
				continue

		center = tuple(point[:2].astype(int))
		size = 5
		cv2.circle(image, center, size, color, thickness=-1)
		is_drawn = True
		
	return image

def draw_points_on_image_parallel(data):
	image, reprojected_points, idx = data
	color = (255,0,0)
	h, w, c = image.shape
	is_drawn = False
	for p_idx, point in enumerate(reprojected_points):
		if point[0] > w or point[0] < 0:
			continue
		if point[1] > h or point[1] < 0:
			continue

		center = tuple(point[:2].astype(int))
		size = 3
		cv2.circle(image, center, size, color, thickness=-1)
		is_drawn = True

	return image

def load_rgb_image(image_path):
	img_texture = cv2.imread(image_path)
	img_texture = cv2.cvtColor(img_texture,cv2.COLOR_BGR2RGB)
	return img_texture


def reproject_on_image(image, cam_ext, cam_int, points, colors=None, use_wrist=False):
	cam_ext_inv = invert_T(cam_ext)
	points_in_cam_coord = (np.einsum('ij,kj->ki',conversions.T2R(cam_ext_inv), points) + conversions.T2p(cam_ext_inv))
	projected_points = np.einsum('ij,kj->ki',cam_int,points_in_cam_coord)
	projected_points /= projected_points[...,2][..., np.newaxis]	

	h, w, c = image.shape

	if colors is not None:
		clr = deepcopy(colors)
		clr = clr[:, ::-1]
		clr *= 255

	wrist = None
	if use_wrist:
		wrist = projected_points[:2, ...]
		projected_points = projected_points[2:, ...]

	for p_idx, point in enumerate(projected_points):
		if point[0] > w or point[0] < 0:
				continue
		if point[1] > h or point[1] < 0:
				continue

		center = tuple(point[:2].astype(int))
		size = 3
		color = (0,0,255)
		cv2.circle(image, center, size, color, thickness=-1)

		if colors is not None:
			color = clr[p_idx]
			color = tuple(color.astype(int).tolist())
			size = 1
			cv2.circle(image, center, size, color, thickness=-1)

	# wrist
	if wrist is None:
		return image
	
	for idx in range(2):
		wrist_point = wrist[idx]
		center = tuple(wrist_point[:2].astype(int))
		color = (0,255,0)
		size = 5
		cv2.circle(image, center, size, color, thickness=-1)

	return image

# write text
def write_info_on_image(image, text_info):
	font = cv2.FONT_HERSHEY_SIMPLEX
	text_position = (10, 30)  # Adjust the values to position the text accordingly
	font_scale = 1
	font_color = (255, 255, 255)  # White color
	line_type = 2
	line_space = 35

	for idx, text_key in enumerate(text_info):
		cur_text = text_info[text_key]
		cur_text_pos = (text_position[0], text_position[1] + idx * line_space)
		cv2.putText(image, 
					str(cur_text), 
					cur_text_pos, 
					font, 
					font_scale, 
					font_color, 
					line_type)

	return image

# following functions is based on image_stream in droid demo
def save_undistorted_image_list(stream, dir):
	stream_list = list(stream)
	for idx in range(len(stream_list)):
		img_idx, img, cam_int = stream_list[idx]
		img_cv_write = img.permute(1, 2, 0).cpu().numpy()
		print("write image")
		cv2.imwrite(f"{dir}/undistort_{idx}.png", img_cv_write.astype(np.uint8))    


def reproject_3d_points(stream, traj_est, pcd_data, reproject_image_path=""):
	stream_list = list(stream)

	cam_ext_p = traj_est[:,:3]
	cam_ext_q = traj_est[:,3:]
	cam_ext_T = conversions.Qp2T(cam_ext_q, cam_ext_p)

	cam_ext_T_inv = invert_T(cam_ext_T)

	pts, clr = pcd_data
	# change to bgr and 0 - 255 value (opencv)
	clr = deepcopy(clr)
	clr = clr[:, ::-1]
	clr *= 255

	for idx in range(len(stream_list)):
		img_idx, img, cam_int = stream_list[idx]
		cam_int = cam_int.numpy()
		img = img[0]

		K = np.zeros((3, 3))
		K[0, 0] = cam_int[0] # fx
		K[1, 1] = cam_int[1] # fy
		K[0, 2] = cam_int[2] # cx
		K[1, 2] = cam_int[3] # cy
		K[2, 2] = 1.0 

		cam_ext_cur = cam_ext_T_inv[idx]
		points_in_cam_coord = (np.einsum('ij,kj->ki',conversions.T2R(cam_ext_cur), pts) + conversions.T2p(cam_ext_cur))
		projected_points = np.einsum('ij,kj->ki',K,points_in_cam_coord)

		# projected_points = (projected_points[:2] / projected_points[...,2])
		projected_points /= projected_points[...,2][..., np.newaxis]

		img_shape = img.shape[1:]
		
		img_cv_write = deepcopy(img.permute(1, 2, 0).cpu().numpy())
		for p_idx, point in enumerate(projected_points):
			if p_idx == 0:
				continue
			# if point[0] > img_shape[0] or point[0] < 0:
			#     continue
			# if point[1] > img_shape[1] or point[1] < 0:
			#     continue
			color = clr[p_idx]
			center = tuple(point[:2].astype(int))
			cv2.circle(img_cv_write, center, 3, (0, 0, 255), -1)
			cv2.circle(img_cv_write, center, 2, tuple(color.astype(int).tolist()), -1)


		cv2.imwrite(f'{reproject_image_path}/reproject_{idx}.png', img_cv_write.astype(np.uint8))
