import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)


import cv2
import pickle
from realdata.realdata_collection import * 
from fairmotion.utils import utils 
import argparse 
from tqdm import tqdm
import re
from realdata.realdata_network_run import *
from realdata.img_utils import *
import subprocess
import constants.path as path_constants

# utils
def is_points_in_bbox(points_2d, bbox, threshold=None):
	# Parsing the bounding box
	x0, y0, width, height = bbox
	x1 = x0 + width
	y1 = y0 + height

	# Check if each point is inside the bounding box
	is_inside = (points_2d[:, 0] >= x0) & (points_2d[:, 0] <= x1) & (points_2d[:, 1] >= y0) & (points_2d[:, 1] <= y1)

	# Count how many points are inside the bounding box
	count_inside = np.sum(is_inside)
	
	if threshold is None:
		threshold = points_2d.shape[0] * 0.8 

	if count_inside > threshold:
		return True 
	else:
		return False	

class HandPosOptim(nn.Module):
	def __init__(self, points_2d, cam_ext, cam_int, init_wrist_transform, points_3d) -> None:
		super(HandPosOptim, self).__init__()

		self.points_2d = torch.tensor(deepcopy(points_2d)).float()
		self.cam_ext_inv = torch.tensor(deepcopy(invert_T(cam_ext))).float()
		self.cam_int = torch.tensor(deepcopy(cam_int)[np.newaxis, np.newaxis, ...]).float()
		self.points_3d = torch.tensor(deepcopy(points_3d)).float()

		init_hand_T = torch.tensor(deepcopy(init_wrist_transform)).float()
		init_hand_rot = transforms.matrix_to_rotation_6d(init_hand_T[...,:3,:3])
		self.hand_rot_6d = nn.Parameter(init_hand_rot)
		self.hand_pos = nn.Parameter(init_hand_T[...,:3,3])

	def forward(self):
		# move hand 3d points
		moved_points_3d = (transforms.rotation_6d_to_matrix(self.hand_rot_6d).unsqueeze(1) @ self.points_3d.unsqueeze(-1))[...,0] + self.hand_pos.unsqueeze(1)

		# apply camera projection
		cam_local_points = (self.cam_ext_inv[..., :3,:3].unsqueeze(1) @ moved_points_3d.unsqueeze(-1))[...,0] + self.cam_ext_inv[..., :3,3].unsqueeze(1)
		reproject_points = (self.cam_int @ cam_local_points.unsqueeze(-1))[...,0]

		# make in homo coordinates
		# reproject_points /= reproject_points[...,2].unsqueeze(-1)
		reproject_points = reproject_points / reproject_points[...,2].unsqueeze(-1)

		# get loss
		reproject_point_2d = reproject_points[...,:2]
		reprojected_diff = torch.abs(reproject_point_2d - self.points_2d) 		

		# get velocity loss
		vel_2d = self.points_2d[1:,...] - self.points_2d[:-1, ...]
		vel_reproj = reproject_point_2d[1:,...] - reproject_point_2d[:-1,...]
		vel_diff = torch.abs(vel_reproj - vel_2d)
		vel_diff = torch.cat((vel_diff[0:1], vel_diff), dim=0) # pad first 

		# embed()
		reproj_diff_root = reprojected_diff[:,0,:]
		reproj_diff_palm = reprojected_diff[:,1:,:]

		reproj_vel_diff_root = vel_diff[:,0,:] 
		reproj_vel_diff_palm = vel_diff[:,1:,:]
		vel_loss = torch.mean(reproj_vel_diff_root)

		reproj_root_loss = torch.mean(reproj_diff_root)
		reproj_palm_loss = torch.mean(reproj_diff_palm)

		reproj_loss = reproj_root_loss + reproj_palm_loss * 1.5 + vel_loss * 0.5

		if reproj_loss.item() < 25:
			reproj_loss = reproj_loss + vel_loss * 0.5 + reproj_root_loss * 0.2

		return reproj_loss

		# if reproj_loss < 15:
		# 	vel_loss = torch.mean(reproj_vel_diff_palm) + torch.mean(reproj_vel_diff_root)
		# 	return reproj_loss + vel_loss

		# return reproj_loss
		# if reproj_root_loss.item() < 20:
		# 	reproj_palm_loss = torch.mean(reproj_diff_palm)
		# 	reproj_loss = reproj_root_loss + reproj_palm_loss * 2
		# 	return reproj_loss
		# elif reproj_root_loss < 30:
		# 	return reproj_root_loss + reproj_palm_loss 
		# else: 
		# 	return reproj_root_loss
		# reprojection_loss = torch.mean(reprojected_diff)
		# return reprojection_loss
	
	def get_current_hand(self):
		return transforms.rotation_6d_to_matrix(self.hand_rot_6d), self.hand_pos


class HandPosConstraint(object):
	def __init__(self, args) -> None:
		self.args = args

		if args.reload_rdc:
			self.load_rdc()
		
		else:
			load_result = parse_realdata_network_output(args=args)
			self.camera_head_signal = load_result['head_signal']
			self.pcd_points = np.asarray(load_result['pcd'].points)
			self.pcd_colors = np.asarray(load_result['pcd'].colors)
			self.motion = load_result['motion']['watch_cam']

		# hand related info
		self.hand_label = ['left', 'right']
		self.palm_joint_idx = [0, 5, 9, 13, 17]
		self.cam_intrinsics = np.eye(3)
		# hardcoding - assuming intrinsic is fixed / later fix...
		self.cam_intrinsics[0,0], self.cam_intrinsics[1,1], self.cam_intrinsics[0,2], self.cam_intrinsics[1,2] = 283.5658, 283.0503, 295.5000, 166.0000

	def load_rdc(self):
		self.rdc = RealDataCollection2(args.realdata_folder)
		self.rdc.load()
		self.camera_head_signal = rdc.camera_head_signal
	
	### Optimization related functions
	def load_and_optim(self):
		self.load_preprocess_result()
		self.optim_wrist()

	def load_preprocess_result(self):

		# preprocess_path = f"/data/realdata/{self.args.realdata_folder}/hand/"
		preprocess_path = os.path.join(path_constants.DATA_PATH, f"{self.args.realdata_folder}/hand/")
		self.points_2d_both = {}
		self.points_3d_both = {}
		self.cam_ext_both = {}
		self.idx_both = {}
		self.wrist_T_both = {}

		# for reprojection & image visualization
		self.image_filename_both = {}
		for hand_label in self.hand_label:
			# to save idx for later visualization & use
			self.points_2d_both[hand_label] = []
			self.points_3d_both[hand_label] = []
			self.cam_ext_both[hand_label] = []
			self.wrist_T_both[hand_label] = []
			self.idx_both[hand_label] = []
			self.image_filename_both[hand_label] = []

			preprocess_dict_path = os.path.join(preprocess_path, f"{hand_label}_mp_handmocap.pkl")
			if not os.path.exists(preprocess_dict_path):
				print(f"{preprocess_dict_path} does not exist!")
				continue
			preprocess_result = pickle.load(open(preprocess_dict_path, "rb"))

			for idx in tqdm(preprocess_result): # index keys
				joint_pos_2d = preprocess_result[idx]['2d_joint_pos'][self.palm_joint_idx,:2]
				if 'pred_joints_smpl' not in preprocess_result[idx]:
					# print(f"{idx} handmocap result not found!")
					continue
				joint_pos_3d = preprocess_result[idx]['pred_joints_smpl'][self.palm_joint_idx, :]
				cam_ext = preprocess_result[idx]['cam_ext']

				self.points_2d_both[hand_label].append(joint_pos_2d)
				self.points_3d_both[hand_label].append(joint_pos_3d)
				self.cam_ext_both[hand_label].append(cam_ext)
				self.idx_both[hand_label].append(idx)
				self.image_filename_both[hand_label].append(preprocess_result[idx]['filename'])

				# load wrist pose
				wrist_jointname = "LeftHand" if 'left' in hand_label else "RightHand"
				wrist_pos = self.motion.poses[idx].get_transform(key=wrist_jointname, local=False)
				self.wrist_T_both[hand_label].append(wrist_pos)

			# after index loop
			self.points_2d_both[hand_label] = np.array(self.points_2d_both[hand_label])
			self.points_3d_both[hand_label] = np.array(self.points_3d_both[hand_label])
			self.cam_ext_both[hand_label] = np.array(self.cam_ext_both[hand_label])
			self.wrist_T_both[hand_label] = np.array(self.wrist_T_both[hand_label])


	def optim_wrist(self):
		# load
		self.module_both = {}
		self.optimizer_both = {}
		self.optim_result_both = {}
		torch.autograd.set_detect_anomaly(True)
		for hand_label in self.hand_label:
			self.module_both[hand_label] = HandPosOptim(points_2d=self.points_2d_both[hand_label], \
						   								cam_ext=self.cam_ext_both[hand_label], \
														cam_int=self.cam_intrinsics, \
														init_wrist_transform=self.wrist_T_both[hand_label], \
														points_3d=self.points_3d_both[hand_label]  
														)
			self.optimizer_both[hand_label] = optim.Adam(self.module_both[hand_label].parameters(), lr=0.1, betas=[0.92, 0.85])

			for i in range(5000):
				loss = self.module_both[hand_label].forward()
				if loss.item() < 15.2: # early stopping 
					break
				print(f"loss: {loss.item()} step: {i} hand: {hand_label}")
				self.optimizer_both[hand_label].zero_grad()
				loss.backward()
				self.optimizer_both[hand_label].step()

			print("\n")

			optimized_wrist_rot, optimized_wrist_pos = self.module_both[hand_label].get_current_hand()
			optimized_wrist_rot = optimized_wrist_rot.detach().cpu().numpy()
			optimized_wrist_pos = optimized_wrist_pos.detach().cpu().numpy()

			optimized_wrist_T = conversions.Rp2T(optimized_wrist_rot, optimized_wrist_pos)
			joint_3d_pos = self.points_3d_both[hand_label]

			optimized_joint_3d_pos = optimized_wrist_T[:, np.newaxis, ...] @ conversions.p2T(joint_3d_pos)
			optimized_joint_3d_pos = conversions.T2p(optimized_joint_3d_pos)

			result = {}
			result['optim_wrist'] = optimized_wrist_T
			result['optim_joint_3d'] = optimized_joint_3d_pos 
			result['idx'] = self.idx_both[hand_label]

			self.optim_result_both[hand_label] = result

			# (temp) visualize to check result - later save to img
			num_frame, num_palm_joint, _ = optimized_joint_3d_pos.shape
			for i in range(num_frame):
				image_filename = self.image_filename_both[hand_label][i]
				image = cv2.imread(image_filename)

				cam_ext_cur_frame = self.cam_ext_both[hand_label][i]
				optimized_joint_3d_pos_cur_frame = optimized_joint_3d_pos[i]

				image = reproject_on_image(image=image, cam_ext=cam_ext_cur_frame, cam_int=self.cam_intrinsics, points=optimized_joint_3d_pos_cur_frame)

				# get text info
				text_info = {}
				text_info['hand_label'] = hand_label
				text_info['frame'] = self.idx_both[hand_label][i]
				image = write_info_on_image(image=image, text_info=text_info)

				cv2.imshow('image', image)
				cv2.waitKey(0)
		
		# result_save_path = f"/data/realdata/{self.args.realdata_folder}/hand/optim_result.pkl"
		result_save_path = os.path.join(path_constants.DATA_PATH, f"{self.args.realdata_folder}/hand/optim_result.pkl")
		pickle.dump(self.optim_result_both, open(result_save_path, "wb"))
		print("finished saving optim result!")

	### Preprocess related functions
	def load_mediapipe(self):
		import mediapipe as mp
		self.mp_hands = mp.solutions.hands
		self.mp_drawing = mp.solutions.drawing_utils

	def load_handmocap(self):
		# add handmocap (from frankmocap) path in docker 
		handmocap_path = '/root/Archive/frankmocap'
		sys.path.append(handmocap_path)
		sys.path.append("/root/Archive/frankmocap/detectors/body_pose_estimator")
		from handmocap.hand_mocap_api import HandMocap
		from handmocap.hand_bbox_detector import HandBboxDetector

		checkpoint_hand_path = os.path.join(handmocap_path, "extra_data/hand_module/pretrained_weights/pose_shape_best.pth")
		smpl_path = os.path.join(handmocap_path, "extra_data/smpl/")

		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		assert torch.cuda.is_available(), "Current version only supports GPU"
		self.bbox_detector =  HandBboxDetector('ego_centric', self.device)
		self.hand_mocap = HandMocap(checkpoint_hand_path, smpl_path, device = "cpu")


	# preprocess with handmocap (frankmocap)
	def handmocap_with_mp_data(self):
		if sys.version_info < (3, 8):
			import pickle5 as pickle 
		self.load_handmocap()
		mp_path = os.path.join(path_constants.DATA_PATH, f"{self.args.realdata_folder}/hand/")

		undistort_image_filenames = self.camera_head_signal.mocap_img_list

		mp_dict_by_hand = {}
		
		# load hand preprocessing (2d via mediapipe) result
		for hand_label in self.hand_label:
			mp_hand_dict_path = os.path.join(mp_path, f"{hand_label}_mp.pkl")
			if not os.path.exists(mp_hand_dict_path):
				print(f"{mp_hand_dict_path} does not exist!")
				continue
			mp_result = pickle.load(open(mp_hand_dict_path, "rb"))

			for img_idx in tqdm(mp_result):
				mp_landmark = mp_result[img_idx]['2d_joint_pos']
				img_filename = undistort_image_filenames[img_idx]
				img_original_bgr = cv2.imread(filename=img_filename)
				detect_output = self.bbox_detector.detect_hand_bbox(img_original_bgr.copy())
				body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = detect_output

				if len(hand_bbox_list) < 1: # hand not found in frankmocap
					continue 

				# run frankmocap
				pred_output_list = self.hand_mocap.regress(img_original_bgr, hand_bbox_list, add_margin=True)
				pred_result = pred_output_list[0] # only hands, not body

				for hand_key in pred_result:
					# get bbox 
					hand_bbox = hand_bbox_list[0][hand_key]
					if hand_bbox is None:
						continue
					# check mp landmarks are in the bounding box 
					is_in_bbox = is_points_in_bbox(points_2d=mp_landmark[:,:2], bbox=hand_bbox)
					# if in, save result 
					if is_in_bbox:
						mp_result[img_idx]['hand_bbox'] = hand_bbox
						mp_result[img_idx].update(pred_result[hand_key])

				mp_dict_by_hand[hand_label] = deepcopy(mp_result)

			# save frankmocap added result dict

		mp_result_save_dir = os.path.join(path_constants.DATA_PATH, f"{args.realdata_folder}/hand/")
		# mp_result_save_dir = f"/data/realdata/{args.realdata_folder}/hand/"
		utils.create_dir_if_absent(mp_result_save_dir)

		for hand_label in mp_dict_by_hand:
			print(f"finished saving {hand_label} in handmocap_with_mp_data")
			mp_hand_path = os.path.join(mp_result_save_dir, f"{hand_label}_mp_handmocap.pkl")
			pickle.dump(mp_dict_by_hand[hand_label], open(mp_hand_path, "wb"))

	# preprocess with mediapipe		
	def mp_preprocess(self):
		self.load_mediapipe()

		undistort_image_filenames = self.camera_head_signal.mocap_img_list
		# undistort_image_filenames = ["/data/realdata/0902_2_short/sfm/undistorted_images/undistort_05612.png"]
		
		result_dict_by_hand = {}
		result_dict_by_hand['right'] = {}
		result_dict_by_hand['left'] = {}

		with self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
			for filename in tqdm(undistort_image_filenames):
				if not (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')):
					continue

				# match = re.search(r'undistort_(\d+).png', filename)
				# number = int(match.group(1))

				# if number not in [7070, 7359]: # just for testing
				# 	continue

				# image_path = os.path.join(directory, filename)
				image = cv2.imread(filename=filename)
				image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				results = hands.process(image_rgb)

				h, w, c = image.shape 
				
				if results.multi_hand_landmarks:
					for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
						score = results.multi_handedness[idx].classification[0].score
						hand_label = results.multi_handedness[idx].classification[0].label.lower()
						img_idx = undistort_image_filenames.index(filename)
						cam_ext = self.camera_head_signal.mocap_cam[img_idx] # save camera extrinsics also
						# extract landmark
						landmark = hand_landmarks.landmark
						# convert to numpy array
						landmark_np = [[lm.x*w, lm.y*h, lm.z] for lm in landmark]
						landmark_np = np.array(landmark_np).astype(np.float32)
						hand_info = {
							'filename': filename,
							'2d_joint_pos': landmark_np,
							'cam_ext': cam_ext
						}
						# add to result dict
						result_dict_by_hand[hand_label][img_idx] = hand_info
		
		# done
		# mp_result_save_dir = f"/data/realdata/{args.realdata_folder}/hand/"
		mp_result_save_dir = os.path.join(path_constants.DATA_PATH, f"{args.realdata_folder}/hand/")

		utils.create_dir_if_absent(mp_result_save_dir)

		for hand_label in result_dict_by_hand:
			print(f"finished saving {hand_label} in mp_preprocess")
			mp_hand_path = os.path.join(mp_result_save_dir, f"{hand_label}_mp.pkl")
			pickle.dump(result_dict_by_hand[hand_label], open(mp_hand_path, "wb"))
	

	def visualize_wrist_reprojection(self):
		undistort_image_filenames = self.camera_head_signal.mocap_img_list

		# select parts that we want
		reproject_img_range_min, reproject_img_range_max = 680, 690 # subject to change
		wrist_joints = ["LeftHand", "RightHand"]

		# create folder to save
		save_path = os.path.join(path_constants.DATA_PATH,f"{self.args.realdata_folder}/hand_vis/wrist_reprojection_{reproject_img_range_min}_{reproject_img_range_max}/")
		# save_path = f"/data/realdata/{self.args.realdata_folder}/hand_vis/wrist_reprojection_{reproject_img_range_min}_{reproject_img_range_max}/"

		utils.create_dir_if_absent(save_path)

		for img_idx, image_filename in enumerate(undistort_image_filenames):
			if img_idx < reproject_img_range_min or img_idx > reproject_img_range_max:
				continue 
			if img_idx == reproject_img_range_min:
				start_number = int(re.search(r'_(\d+)\.', image_filename).group(1))
				
			image = cv2.imread(image_filename)
		
			wrist_points = []
			for joint in wrist_joints:
				wrist_pose = self.motion.poses[img_idx].get_transform(key=joint, local=False)[:3,3]
				wrist_points.append(wrist_pose)
			
			# get pcd points close to the wrist
			distances_to_left = np.linalg.norm(self.pcd_points - wrist_points[0], axis=1)
			distances_to_right = np.linalg.norm(self.pcd_points - wrist_points[1], axis=1)

			# Find all points within a distance of 0.4 from either 3D point
			mask = (distances_to_left < 0.4) | (distances_to_right < 0.4)
			pcd_points_to_reproject = self.pcd_points[mask]
			pcd_point_colors = self.pcd_colors[mask]

			# pcd_points_to_reproject = self.pcd_points[(distances_to_left < 0.5) | (distances_to_right < 0.5)]
			points_to_reproject = np.concatenate((wrist_points, pcd_points_to_reproject), axis=0)

			cam_ext_cur = self.camera_head_signal.mocap_cam[img_idx]
			
			image = reproject_on_image(image=image, cam_ext=cam_ext_cur, cam_int=self.cam_intrinsics, points=points_to_reproject, colors=pcd_point_colors, use_wrist=True)
			
			cv2.imshow('image', image)
			cv2.waitKey(0)
			
			# save image in folder
			reprojected_img_filepath = os.path.join(save_path, image_filename.split('/')[-1])
			cv2.imwrite(reprojected_img_filepath, image)

		# make a video using ffmpeg 
		video_dir = os.path.join(path_constants.DATA_PATH, f"{self.args.realdata_folder}/hand_vis/video/")

		# video_dir = f"/data/realdata/{self.args.realdata_folder}/hand_vis/video/"
		utils.create_dir_if_absent(video_dir)

		ffmpeg_command = f"ffmpeg -framerate 30 -start_number {start_number} -i {save_path}undistort_%05d.png -c:v libx264 -crf 20 -pix_fmt yuv420p {video_dir}wrist_reprojection_{reproject_img_range_min}_{reproject_img_range_max}.mp4"
		result = subprocess.run(ffmpeg_command, shell=True, check=True)
		
		# if result.returncode != 0:
		# 	print(f"Command failed with exit code {result.returncode}")
		# 	print(f"Stdout: {result.stdout.decode('utf-8')}")
		# 	print(f"Stderr: {result.stderr.decode('utf-8')}")
		# else:
		# 	print("Command succeeded")
		# return 

	def visualize_preprocess_result(self):
		mp_path = os.path.join(path_constants.DATA_PATH, f"{self.args.realdata_folder}/hand/")
		
		undistort_image_filenames = self.camera_head_signal.mocap_img_list
		
		# load hand preprocessing (2d via mediapipe) result
		for hand_label in self.hand_label:
			mp_hand_dict_path = os.path.join(mp_path, f"{hand_label}_mp_handmocap.pkl")
			if not os.path.exists(mp_hand_dict_path):
				print(f"{mp_hand_dict_path} does not exist!")
				continue

			mp_result = pickle.load(open(mp_hand_dict_path, "rb"))

			# create dir to save images
			save_dir = os.path.join(mp_path, f"{hand_label}_img/")
			utils.create_dir_if_absent(save_dir)

			for img_idx in mp_result:
				img_filename = undistort_image_filenames[img_idx]
				image = cv2.imread(filename=img_filename)

				# load data 
				mp_landmark = mp_result[img_idx]['2d_joint_pos']
				handmocap_bbox = mp_result[img_idx]['hand_bbox']
				handmocap_joints_img = mp_result[img_idx]['pred_joints_img']

				# draw 
				num_joints, _ = mp_landmark.shape
				for joint_idx in range(num_joints):
					if joint_idx in [0, 5, 9, 13, 17]:
						mp_point, handmocap_point = mp_landmark[joint_idx], handmocap_joints_img[joint_idx]
						cv2.circle(image, (int(mp_point[0]), int(mp_point[1])), 2, (255, 0, 0), -1)  # mp: blue
						cv2.circle(image, (int(handmocap_point[0]), int(handmocap_point[1])), 2, (0, 0, 255), -1)  # mp: blue
						# draw bbox 
						x0, y0, w, h = handmocap_bbox.astype(int)
						cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), (0,255,0), 2)

				# save img 
				reprojected_img_filepath = os.path.join(save_dir, img_filename.split('/')[-1])
				print(f"writing image {reprojected_img_filepath}...")
				cv2.imshow('image', image)
				cv2.waitKey(0)
				cv2.imwrite(reprojected_img_filepath, image.astype(np.uint8))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument("--realdata_folder", type=str)
	parser.add_argument("--realdata-folder", type=str, default="", required=True)
	parser.add_argument("--test-name", type=str, default="")
	parser.add_argument("--bm-path",default="../data/smpl_models/smplx/SMPLX_NEUTRAL.npz")
	parser.add_argument("--command-type", choices=['mp', 'handmocap', 'vis_preprocess', 'vis_reproject', 'optim'])
	# parser.add_argument("--use_mp", default=False)
	parser.add_argument("--reload_rdc", default=False)
	args = parser.parse_args()

	hpc = HandPosConstraint(args=args)

	if args.command_type == "mp":
		hpc.mp_preprocess()
	elif args.command_type == "handmocap":
		hpc.handmocap_with_mp_data()
	elif args.command_type == "vis_preprocess":
		hpc.visualize_preprocess_result()
	elif args.command_type == "vis_reproject":
		hpc.visualize_wrist_reprojection()
	elif args.command_type == "optim":
		hpc.load_and_optim()
		
	# if args.use_mp:
	# 	hpc.mp_preprocess()
	# else:
	# 	hpc.handmocap_with_mp_data()
		# hpc.visualize_preprocess_result()
		# hpc.visualize_wrist_reprojection()
