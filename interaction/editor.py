import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
import argparse
from fairmotion.core import motion as motion_class
from fairmotion.ops import conversions, math, motion as motion_ops
from IPython import embed 
from realdata.realdata_collection import *
from realdata.realdata_network_run import *
import pickle 
import cv2
import constants.motion_data as motion_constants
import constants.path as path_constants
from imu2body.imu import *
from autoencoder_v2.run import * 
from interaction.autoencoder_optimizer import * 
from fairmotion.ops.math import * 
from realdata.footcleanup import *

logging.basicConfig(
	format="[%(asctime)s] %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	level=logging.INFO,
)


def postprocess_edit_result(dir_name, draw_reproj=True):
	# open file 
	result_dict = {}

	result_pkl_path = os.path.join(path_constants.BASE_PATH, f"results/edit/{dir_name}/result.pkl")
	with open(result_pkl_path, "rb") as file:
		edit_result_dict = pickle.load(file)
	
	config = edit_result_dict['config']
	result_dict['rdc_folder'] = config['rdc_folder']

	result_dict['config'] = config

	# blend
	for start_frame in edit_result_dict['edited_motion_dict']:
		target_pose = edit_result_dict['motion']['edited'].poses[start_frame]

		for i_before in range(8, 0, -1):
			weight = i_before * (1/8.0)
			interpolated_pose = motion_class.Pose.interpolate(deepcopy(edit_result_dict['motion']['edited'].poses[start_frame-i_before]), target_pose, 1-weight)
			edit_result_dict['motion']['edited'].poses[start_frame-i_before] = interpolated_pose


	result_dict['motion'] = {}
	result_dict['motion']['initial'] = edit_result_dict['motion']['initial']
	result_dict['motion']['edited'] = edit_result_dict['motion']['edited']
	if 'skel_with_offset' in edit_result_dict:
		result_dict['skel_with_offset'] = edit_result_dict['skel_with_offset']

	result_dict['edit_info'] = edit_result_dict['edited_motion_dict']

	result_dict['reproj_dict'] = edit_result_dict['reproj']

	# load rdc results
	rdc_folder = config['rdc_folder']
	# first read signals
	cam_filepath = os.path.join(path_constants.DATA_PATH, f"{rdc_folder}/cam_signal.pkl")
	imu_filepath = os.path.join(path_constants.DATA_PATH, f"{rdc_folder}/imu_signal.pkl")

	with open(cam_filepath, "rb") as cam_file:
		cam_data = pickle.load(cam_file)
	with open(imu_filepath, "rb") as imu_file:
		imu_data = pickle.load(imu_file)
	
	cam_data['camera_signal'].load_pcd_result()
	camera_head_signal = cam_data['camera_signal']
	result_dict['imu_signal'] = imu_data
	result_dict['head_signal'] = camera_head_signal
	result_dict['pcd'] = camera_head_signal.pcd 
	result_dict['cam_ext'] = camera_head_signal.mocap_cam
	
	# load image texture
	img_texture_list = utils.run_parallel(cv2.imread, result_dict['head_signal'].mocap_img_list)
	result_dict['img_texture'] = img_texture_list

	if draw_reproj:

		for frame in edit_result_dict['reproj']:
			detected = edit_result_dict['reproj'][frame]['target_2d']

			img = result_dict['img_texture'][frame]
			cv2.circle(img, (int(detected[0]), int(detected[1])), 7, (0, 255, 0), -1)  # mp: green

			result_dict['img_texture'][frame] = img

		# draw reprojection on the texture
		start_frame = list(edit_result_dict['reproj'].keys())[2]
		cam_ext = result_dict['head_signal'].mocap_cam
		cam_int = result_dict['head_signal'].intrinsics 

		reproj_color_dict = {
			'initial': (0, 0, 255),
			'edited': (255, 160, 24),
			'gt':(132, 132, 12)
		}
	
		# only wrist
		if frame in edit_result_dict['reproj']:
			lr = edit_result_dict['reproj'][frame]['joint_idx'] - 20
		initial_point = edit_result_dict['wrist_reprojected']['initial'][frame,lr]
		edited_point = edit_result_dict['wrist_reprojected']['edited'][frame,lr]

		cv2.circle(img, (int(initial_point[0]), int(initial_point[1])), 5, (0, 0, 255), -1)  # initial: red
		cv2.circle(img, (int(edited_point[0]), int(edited_point[1])), 5, (255, 160, 24), -1)  # edited: blue

	return result_dict

def read_img(img_filename):
	image = cv2.imread(img_filename)
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image_rgb

def read_img_flag(img_filename, flag):
	return cv2.imread(img_filename, flag)

class MotionEditor(object):
	def __init__(self, args):
		self.args = args
		result_dict = parse_realdata_network_output(args=args)

		self.data_folder = args.rdc_folder
		self.result_dict = result_dict

		self.initial_motion = result_dict['motion']['watch_cam']	
		self.initial_joint_local_T = self.initial_motion.to_matrix(local=True)
		self.initial_joint_global_T = self.initial_motion.to_matrix(local=False) # [F, J, 3]
		
		self.contact_label = result_dict['motion_contact_label']['watch_cam']
		self.contact_dict = result_dict['contact']
		self.camera_head_signal = result_dict['head_signal']
		self.imu_signal = result_dict['imu_signal']

		# edited motion (placeholder)
		self.edited_motion = deepcopy(self.initial_motion)
		self.edited_motion_dict = {}

		# constraint
		self.reprojected_wrist_pos = {}
		self.reprojected_arm_pos = {}

		self.constraints = {}
		self.loss = {}
		self.normalized_loss = {}
		self.loss_mean = {}
		self.loss_std = {}

		# optimizer
		self.autoencoder_optimizer = None 

		# optim numbers
		self.discard_threshold = 0.45 

		return
	
	def load_img_list(self):
		mocap_img_list = self.result_dict['head_signal'].mocap_img_list
		img_texture_list = utils.run_parallel(cv2.imread, mocap_img_list, num_cpus=24)
	
		self.result_dict['img_texture'] = img_texture_list
	
	def normalize_loss(self, loss):
		has_nan = np.isnan(loss).any()
		loss_mean = np.mean(loss, axis=0) if not has_nan else np.nanmean(loss, axis=0)
		loss_std = np.std(loss, axis=0) if not has_nan else np.nanstd(loss, axis=0)

		normalized_loss = (loss - loss_mean) / (loss_std + constants.EPSILON)
		return normalized_loss, loss_mean, loss_std

	def get_contact_loss(self):    
		foot_pos = self.initial_joint_global_T[:,motion_constants.foot_joint_idx,:3,3]
		foot_pos = np.transpose(foot_pos, (1, 0, 2))
		foot_vel = foot_pos[1:] - foot_pos[:-1]
		foot_vel = np.concatenate([foot_vel[0:1], foot_vel], axis=0) # append first frame
		foot_vel_norm = np.linalg.norm(foot_vel, axis=-1)

		self.loss['contact'] = foot_vel_norm * self.contact_label # [F, 2]
		self.normalized_loss['contact'], self.loss_mean['contact'], self.loss_std['contact'] = self.normalize_loss(loss=self.loss['contact'])


	# reprojection related
	def reproject_wrist(self, type="initial"):
		cam_ext = self.camera_head_signal.mocap_cam 
		cam_int = self.camera_head_signal.intrinsics 

		if type == "initial":
			wrist_pos = self.initial_joint_global_T[:,motion_constants.imu_hand_joint_idx,:3,3]
		else:
			wrist_pos = self.edited_motion_global_T[:,motion_constants.imu_hand_joint_idx,:3,3]

		wrist_pos = np.transpose(wrist_pos, (1,0,2))

		# reproject 
		cam_ext_inv = invert_T(cam_ext)
		points_in_cam_coord = np.einsum('ijk, ilk->ilj',conversions.T2R(cam_ext_inv), wrist_pos) + conversions.T2p(cam_ext_inv)[:,np.newaxis,:]
		projected_points = np.einsum('ij, klj->kli', cam_int, points_in_cam_coord)
		projected_points /= projected_points[...,2][...,np.newaxis] 

		self.reprojected_wrist_pos[type] = projected_points
	
	
	def load_yolo_model(self):
		from ultralytics import YOLO # yolov8 
		if os.path.exists('./yolov8n-pose.pt'):
			self.yolo_pos_model = YOLO('./yolov8n-pose.pt')
		else:
			self.yolo_pos_model = YOLO('yolov8n-pose.pt')
	
	# def get_2d_wrist_yolov8(self):
	# 	self.load_yolo_model()
	# 	undistort_image_filenames = self.camera_head_signal.mocap_img_list
	# 	img_bgr_list = utils.run_parallel(cv2.imread, undistort_image_filenames)

	# 	detected = []
	# 	for frame_idx, file in enumerate(tqdm(undistort_image_filenames)):
	# 		result = self.yolo_pos_model(img_bgr_list[frame_idx])
	# 		if result[0].keypoints.conf is not None:
	# 			detected.append(file)
		
	# 	print("--------------------")
	# 	for d in detected:
	# 		print(d)

	# 2d keypoint using mediapipe
	def get_2d_wrist_mp(self):
		import mediapipe as mp
		self.mp_hands = mp.solutions.hands
		undistort_image_filenames = self.camera_head_signal.mocap_img_list

		mp_result_list = []

		img_rgb_list = utils.run_parallel(read_img, undistort_image_filenames)

		with self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
			for frame_idx, filename in enumerate(tqdm(undistort_image_filenames)):
				if not (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')):
					continue

				image_rgb = img_rgb_list[frame_idx]
				results = hands.process(image_rgb)

				h, w, c = image_rgb.shape 
				
				if results.multi_hand_landmarks:
					for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
						score = results.multi_handedness[idx].classification[0].score

						hand_label = results.multi_handedness[idx].classification[0].label.lower()
						landmark = hand_landmarks.landmark
						landmark_np = [[lm.x*w, lm.y*h, lm.z] for lm in landmark]
						landmark_np = np.array(landmark_np).astype(np.float32)

						mp_info = {
							'frame': frame_idx,
							'2d_wrist_pos': landmark_np[0,:2], # save wrist only
							'hand_label': hand_label
						}
						mp_result_list.append(mp_info)

		mp_result_path = f"{path_constants.DATA_PATH}/{self.data_folder}/mp_wrist.pkl"
		with open(mp_result_path, "wb") as file:
			pickle.dump(mp_result_list, file=file, protocol=pickle.HIGHEST_PROTOCOL)


	def infilling_2d(self, mp_result_list):
		mp_result_dict_frames = {}
		undistort_image_filenames = self.camera_head_signal.mocap_img_list

		img_in_greyscale = utils.run_parallel(read_img_flag, undistort_image_filenames, num_cpus=24, flag=cv2.IMREAD_GRAYSCALE)

		for mp_result in mp_result_list:
			frame = mp_result['frame']
			mp_result['infilled'] = False
			mp_result_dict_frames[frame] = mp_result
			mp_result_dict_frames[frame]['img_filename'] = undistort_image_filenames[frame]

		# search for points where detection "starts"
		hand_detection_start = {}
		for frame in mp_result_dict_frames:
			prev_frame = frame-1
			if prev_frame not in mp_result_dict_frames:
				hand_detection_start[frame] = mp_result_dict_frames[frame]

		# infill by optical flow
		infilled_frame = []
		for start_frame in hand_detection_start:
			cur_frame = start_frame
			prev_frame = cur_frame-1
			tracked_point = hand_detection_start[start_frame]['2d_wrist_pos']
			hand_label = hand_detection_start[start_frame]['hand_label']

			while True:
				if prev_frame in mp_result_dict_frames:
					break

				cur_img = img_in_greyscale[cur_frame]
				prev_img = img_in_greyscale[prev_frame]
				new_points, status, err = cv2.calcOpticalFlowPyrLK(cur_img, prev_img, tracked_point.reshape(-1, 1, 2), None)

				if err[0,0] > 5.5: # stop tracking when errornaeous tracking starts
					break

				# if optical flow found 
				if status[0][0]:

					mp_result_dict_frames[prev_frame] = {
						'frame': prev_frame,
						'2d_wrist_pos': new_points[0,0],
						'hand_label': hand_label,
						'infilled':True
					}
					mp_result_list.append(mp_result_dict_frames[prev_frame])
					infilled_frame.append(prev_frame)
					cur_frame = prev_frame
					prev_frame = cur_frame-1
				
				# if not found, break
				else:
					break
			
		# (for vis to check optical flow is working)
		infill_img_save_dir = os.path.join(path_constants.DATA_PATH, self.args.rdc_folder, "infill_images_color")
		utils.create_dir_if_absent(infill_img_save_dir)
		for mp_result in mp_result_list:
			img = img_in_greyscale[mp_result['frame']]
			img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
			point = mp_result['2d_wrist_pos']
			
			if mp_result['infilled']:
				color = (0,0,255)
			else:
				color = (0,255,0)

			cv2.circle(img_rgb, (int(point[0]), int(point[1])), 5, color, -1)		
			cv2.imwrite(f"{infill_img_save_dir}/{mp_result['frame']}_infill.png", img_rgb)
   
		return mp_result_list


	def get_reprojection_loss(self):
		self.reproject_wrist() # reproject wrist first

		mp_result_path = f"{path_constants.DATA_PATH}/{self.data_folder}/mp_wrist.pkl"

		if not os.path.exists(mp_result_path):
			self.get_2d_wrist_mp()
		
		# load mp result
		with open(mp_result_path, "rb") as file:
			mp_result_list = pickle.load(file=file)

		# infill using optical flow
		mp_result_list = self.infilling_2d(mp_result_list=mp_result_list)

		# loop through the list and save losses
		frame_num = self.initial_joint_global_T.shape[0]
		reproject_loss = np.full((frame_num, 2), np.nan) # both wrist 

		# constraint for reproject
		self.constraints['reproj']  = {}

		for mp_result in mp_result_list:
			frame = mp_result['frame']
			wrist_2d = mp_result['2d_wrist_pos']
			reprojected_both_wrist = self.reprojected_wrist_pos['initial'][frame,:,:2] # remove z

			diff = np.abs(reprojected_both_wrist - wrist_2d)
			norms = np.linalg.norm(diff, axis=1)
			lr = np.argmin(norms)

			mp_result['hand_label'] = lr
			reproject_loss[frame,lr] = norms[lr]

			# update constraint
			self.constraints['reproj'][frame] = {
				'target_2d': wrist_2d,
				'joint_idx': motion_constants.imu_hand_joint_idx[lr],
				'cam_ext': self.camera_head_signal.mocap_cam[frame],
			}


		self.loss['reproject'] = reproject_loss
		self.normalized_loss['reproject'], self.loss_mean['reproject'], self.loss_std['reproject'] = self.normalize_loss(reproject_loss)

	def update_reprojection(self):
		self.reproject_wrist(type="edited")
	

	def get_losses(self):
		self.get_contact_loss()
		self.get_reprojection_loss()


	# autoencoder optimizer related
	def load_autoencoder_optimizer(self):
		# first load autoencoder
		self.args.mode == "custom"
		self.args.test_name = self.args.ae_test_name
		logging.info(f"Start loading autoencoder in editor...")
		ae_network = AutoEncoder(args=self.args) # TODO but need to add mean and std here! fix autoencoder loading function
		self.autoencoder_optimizer = AutoEncoderOptim(ae_network=ae_network)
		logging.info(f"Start loading autoencoder in editor...")
		
		# set loss std
		self.autoencoder_optimizer.set_loss_std(self.loss_std)


	def edit_motion_loop(self):
		self.debug = False

		if self.autoencoder_optimizer is None and self.debug is False:
			self.load_autoencoder_optimizer()

		self.autoencoder_window_size = motion_constants.preprocess_window * 2
		self.padding = 5

		start_frame = list(self.constraints['reproj'].keys())[0]
		
		jump = 50
		end_frame = start_frame + jump

		count = 0

		end_break = len(self.initial_motion.poses)
		end_break =  list(self.constraints['reproj'].keys())[-1]
		while True:

			if end_frame >= end_break:
				break 

			# prepare input for latent optimization 
			input_ = {}
			input_['start_frame'] = start_frame 
			input_['global_T'] = deepcopy(self.initial_joint_global_T[start_frame:start_frame+self.autoencoder_window_size,...])
			input_['local_T'] = deepcopy(self.initial_joint_local_T[start_frame:start_frame+self.autoencoder_window_size])
			input_['real_imu_acc'] = deepcopy(self.imu_signal['imu_acc'][start_frame:start_frame+self.autoencoder_window_size])
			input_['real_imu_rot'] = deepcopy(self.imu_signal['imu_rot'][start_frame:start_frame+self.autoencoder_window_size])
			input_['contact_label'] = deepcopy(self.contact_label[start_frame:start_frame+self.autoencoder_window_size])
			input_['reproj_target'] = [(k, v) for k, v in self.constraints['reproj'].items() if start_frame <= k < end_frame]

			# preprocess input for latent optimization
			input_dict = self.preprocess_input(data_dict=input_, debug=self.debug)
			
			if self.debug:
				return input_dict

			result_output_T, result_global_T, loss_record = self.autoencoder_optimizer.latent_optimize(input_data_dict=input_dict)
			if result_output_T is None:
				start_frame = end_frame
				end_frame = start_frame + jump					
				continue

			if loss_record['total'] < self.discard_threshold and loss_record['total'] < loss_record['init']:
				print(f"save motion record")
				self.edited_motion_dict[start_frame] = {}
				self.edited_motion_dict[start_frame]['edited_T'] = result_output_T[0:jump]
				self.edited_motion_dict[start_frame]['initial_T'] = self.initial_joint_local_T[start_frame:start_frame+jump]
				self.edited_motion_dict[start_frame]['loss_record'] = loss_record
				self.edited_motion_dict[start_frame]['frame_range'] = [start_frame, start_frame+jump]
				
				self.initial_joint_local_T[start_frame:start_frame+jump] = result_output_T[0:jump]

			# move on to next 
			start_frame = end_frame
			end_frame = start_frame + jump
			count += 1
			
		self.edited_motion = motion_classes.Motion.from_matrix(self.initial_joint_local_T, skel=deepcopy(self.initial_motion.skel)) 
		self.edited_motion_global_T = self.edited_motion.to_matrix(local=False)
		self.update_reprojection()


	def save_results(self):
		result_dict = {}

		# config 
		config = {}
		config['discard_threshold'] = self.discard_threshold
		config['loss_weight'] = self.autoencoder_optimizer.loss_weight 
		config['network_name'] = self.args.test_name
		config['ae_network_name'] = self.args.ae_test_name
		config['bm_type'] = self.args.bm_type
		config['window_size'] = self.autoencoder_window_size
		config['rdc_folder'] = self.args.rdc_folder

		result_dict['config'] = config 

		# reprojection
		result_dict['reproj'] = self.constraints['reproj']

		# results
		result_dict['motion'] = {}
		result_dict['motion']['initial'] = self.initial_motion
		result_dict['motion']['edited'] = self.edited_motion 
		result_dict['skel_with_offset'] = self.result_dict['skel_with_offset']

		result_dict['edited_motion_dict'] = self.edited_motion_dict
		result_dict['wrist_reprojected'] = self.reprojected_wrist_pos

		# create save path (file io related)
		save_name = self.args.save_name if self.args.save_name != "" else self.args.rdc_folder
		edit_result_dir = os.path.join(path_constants.BASE_PATH, f"results/edit/{save_name}/")
		utils.create_dir_if_absent(edit_result_dir)
		edit_result_save_path = os.path.join(edit_result_dir, f"result.pkl")

		with open(edit_result_save_path, "wb") as file:
			pickle.dump(result_dict, file)
					

	def preprocess_input(self, data_dict, debug=False):
		# adjust height
		min_foot_height = np.min(data_dict['global_T'][:,motion_constants.toe_joints_idx,motion_constants.height_indice])
		original_local_T = data_dict['local_T']

		# adjust height for local_T and global_T (global_T is for reg terms)
		height_adjusted_local_T = deepcopy(original_local_T)
		height_adjusted_local_T[:,0,motion_constants.height_indice,3] -= min_foot_height
		height_adjusted_global_T = deepcopy(data_dict['global_T']) # [90, 22, 3]
		height_adjusted_global_T[...,motion_constants.height_indice,3] -= min_foot_height

		height_adjusted_global_T = height_adjusted_global_T[np.newaxis, ...].astype(dtype=np.float32)
		height_adjusted_local_T = height_adjusted_local_T[np.newaxis, ...].astype(dtype=np.float32)

		# get transformation matrix that maps start frame into (0,-y,0) and (0,0,h)
		v_face_skel = self.initial_motion.skel.v_face
		v_up_env = self.initial_motion.skel.v_up_env

		forward = motion_constants.plane[np.newaxis,...] * (np.einsum('ijk,k->ij',height_adjusted_local_T[...,0,0,:3,:3],v_face_skel))
		facing_rotations = R_from_vectors_tensor(forward, motion_constants.facing_dir_env) # send forward to facing_dir_env
		normalized_root_start_rot = facing_rotations[:,np.newaxis, np.newaxis, ...] @ height_adjusted_local_T[:,0:1,0:1,:3,:3]
		normalized_root_start_pos = np.zeros_like(height_adjusted_local_T[...,0:1,0:1,:3,3])
		normalized_root_start_pos[...,motion_constants.height_indice] = height_adjusted_local_T[:,0:0+1,0:1,motion_constants.height_indice,3] # only height left
		normalized_root_start_T = conversions.Rp2T(normalized_root_start_rot, normalized_root_start_pos)

		ha_to_normalized = normalized_root_start_T @ invert_T(height_adjusted_local_T[:,0:1,0:1,...])
		height_adjusted_local_T[...,0:1,:,:] = ha_to_normalized @ height_adjusted_local_T[...,0:1,:,:] # normalized
		height_adjusted_global_T = ha_to_normalized @ height_adjusted_global_T

		input_local_T = height_adjusted_local_T
		input_global_T = height_adjusted_global_T

		# rotate imu acc signal with the R of the transformation matrix 
		normalized_imu_acc = np.einsum('ijk,lmk->ilmj', ha_to_normalized[:,0,0,:3,:3], data_dict['real_imu_acc'])
		normalized_imu_rot = np.einsum('ijk,lmkn->ilmjn', ha_to_normalized[:,0,0,:3,:3], data_dict['real_imu_rot'])[0]

		# transform cam_ext of the reproj_target dict for reprojection 
		reproject_target_list = []
		for frame, re_dict in data_dict['reproj_target']:
			# first adjust height of cam ext (based on min_foot_height)
			# and then do ha_to_normalized @ height_adjusted_cam_ext
			ha_cam_ext = deepcopy(re_dict['cam_ext'])
			ha_cam_ext[motion_constants.height_indice,3] -= min_foot_height
			normalized_cam_ext = ha_to_normalized[0,0,0] @ ha_cam_ext
			# update reprojection target dict
			re_dict['normalized_cam_ext'] = normalized_cam_ext
			re_dict['frame'] = frame - data_dict['start_frame']
			reproject_target_list.append(re_dict)

		# save transformation matrix (inv) that maps normalized -> original 
		original_local_T = original_local_T[np.newaxis, ...]
		normalized_to_original = original_local_T[:,0:1,0:1,...] @ invert_T(normalized_root_start_T)

		if debug:
			return input_local_T, normalized_imu_acc, reproject_target_list, normalized_to_original, input_global_T[...,:3,3]
	
		# get input_seq which concats and changes into vector [80, 135]
		batch, seq_len, num_joint, _, _ = input_local_T.shape

		joint_rot = T_to_6d_rot(input_local_T)
		joint_rot = joint_rot.reshape(batch, seq_len, -1)
		root_pos = input_local_T[...,0,:3,3]
		c_lr = data_dict['contact_label'][np.newaxis, ...]
		input_seq = np.concatenate((c_lr, root_pos, joint_rot), axis=-1)
		input_seq = input_seq[:,0:self.autoencoder_window_size, ...] # TODO check

		# dict
		preprocess_dict = {}
		preprocess_dict['input_seq'] = input_seq.astype(dtype=np.float32)
		preprocess_dict['imu_acc'] = normalized_imu_acc
		preprocess_dict['imu_rot'] = transforms.matrix_to_rotation_6d(torch.from_numpy(normalized_imu_rot))
		preprocess_dict['reproj_target'] = reproject_target_list
		preprocess_dict['cam_int'] = self.camera_head_signal.intrinsics 
		# for decoding and comparison
		preprocess_dict['normalized_local_T'] = input_local_T # reg terms
		preprocess_dict['contact_label'] = data_dict['contact_label']
		preprocess_dict['global_p'] = input_global_T[...,:3,3]
		preprocess_dict['frame_start_end'] = [0, self.autoencoder_window_size]
		preprocess_dict['root_start'] = normalized_to_original
		preprocess_dict['start_frame'] = data_dict['start_frame']

		return preprocess_dict

# later load visualizer
if __name__ == "__main__":
	parser = argparse.ArgumentParser(
	description="Real Data Visualization"
	)

	# add argparse (file info)
	parser.add_argument("--rdc-folder", type=str, default="", required=True)
	parser.add_argument("--test-name", type=str, default="") # imu2body network 
	parser.add_argument("--ae-test-name", type=str, default="") # ae network name
	parser.add_argument("--mode", type=str, default="custom") # ae mode
	parser.add_argument("--bm-type",default="smplx")
	# parser.add_argument("--load-vis",default=False)
	parser.add_argument("--save-name", type=str, default="")
	parser.add_argument("--cleanup", type=bool, default=True) # cleanup initial motion before sending to editor

	args = parser.parse_args()

	editor = MotionEditor(args=args)

	editor.get_losses()
	editor.edit_motion_loop()
	
	# save results
	editor.save_results()
