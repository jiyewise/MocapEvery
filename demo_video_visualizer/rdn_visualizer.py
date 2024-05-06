import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
from logging import root
import numpy as np
import os
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image

from fairmotion.viz import camera, gl_render, glut_viewer
from fairmotion.data import bvh, asfamc
from fairmotion.ops import conversions, math, motion as motion_ops
from fairmotion.utils import utils
from fairmotion.utils import constants as fairmotion_constants

from IPython import embed
import pickle
from copy import deepcopy
import constants.motion_data as motion_constants
import time
from visualizer.renderables import PointCloud, MeshSequence, Mesh
import cv2
import constants.path as path_constants
from fairmotion.core import motion as motion_class

# for converting into smpl meshes
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import torch 
import trimesh

from pytorch3d.structures import Meshes

from tqdm import tqdm
import subprocess 

# def convert_to_trimesh(vertices, faces):
# 	return trimesh.Trimesh(vertices=vertices, faces=faces, force_mesh=True, process=False)

class RealdataNetworkViewer(glut_viewer.Viewer):

	def __init__(
		self,
		result_dict,
		play_speed=1.0,
		scale=1.0,
		thickness=1.0,
		render_overlay=False,
		hide_origin=False,
		**kwargs,
	):
		self.result_dict = result_dict
		self.render_overlay = render_overlay
		self.hide_origin = hide_origin
		self.camera_mode = "free" # free (mouse), tracking, fixed, keyframe
		self.render_contact_label = 'motion_contact_label' in result_dict.keys()
		self.root_record = []


		# render options
		self.render_motion_type = "all"
		self.render_mesh_dict = {}
		self.pcd_point_size = 3.0
		# glEnable(GL_POINT_SMOOTH)
		# glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

		self.colors = {
			"result": np.array([58, 179, 255, 255]) / 255.0,  # blue
		}


		# load pcd
		self.is_pcd_render_init = False
		self.load_pcd_renderables()

		self.play_speed = play_speed
		self.cur_time = 0.0
		self.play = True
		self.cur_frame = 0

		self.scale = scale
		self.thickness = thickness * 1.5

		self.is_mesh_render_init = False
		self.load_smpl_mesh_related()

		# camera (for demo)
		self.camera_dict = {}
		self.record = False
		self.record_cams = False
		self.read_cams = False
		self.show_pcd = True
		self.cur_floor = 0

		# length of motion
		self.motion_num = len(self.result_dict['motion'])
		super().__init__(**kwargs)

	
	def record_cam_key(self):
		cur_cam = {'pos': self.cam_cur.pos, 'origin': self.cam_cur.origin}
		self.camera_dict[self.cur_frame] = deepcopy(cur_cam)
		
	def load_cam_dict(self, cam_dict):
		self.camera_dict = cam_dict
		self.read_cams = True


	def read_cam_dict_and_set(self):
		from scipy import ndimage
		pos_list = []
		origin_list = []
		for f in self.camera_dict:
			pos_list.append(self.camera_dict[f]['pos'])
			origin_list.append(self.camera_dict[f]['origin'])

		pos_list_filtered = ndimage.uniform_filter1d(np.array(pos_list), size=7, axis=0, mode="nearest")
		origin_list_filtered = ndimage.uniform_filter1d(np.array(origin_list), size=7, axis=0, mode="nearest")

		start = list(self.camera_dict.keys())[0]
		for f in self.camera_dict:
			self.camera_dict[f]['pos'] = pos_list_filtered[f-start]
			self.camera_dict[f]['origin'] = origin_list_filtered[f-start]

		if self.cur_frame not in self.camera_dict:
			return 
		cur_cam_info = self.camera_dict[self.cur_frame]
		pos = cur_cam_info['pos']
		origin = cur_cam_info['origin']
		self.cam_cur.pos = pos 
		self.cam_cur.origin = origin



	def load_smpl_mesh_related(self):
		self.mesh_seq_dict = {}
		self.bm_type = "smplx"

		if self.bm_type == "smplx":
			import imu2body.amass as amass 
		if self.bm_type == "smplh":
			import imu2body_eval.amass_smplh as amass

		bm_path = motion_constants.BM_PATH if self.bm_type == "smplx" else motion_constants.SMPLH_BM_PATH
		self.body_model = amass.load_body_model(bm_path=bm_path)

		for otype in self.result_dict['motion']:
			if otype == "gt":
				continue
			motion = self.result_dict['motion'][otype]
			print(otype)
			vertices_seq, faces = amass.create_mesh_from_amass_fairmotion(motion=motion, bm=self.body_model)

			num_seq, _, _ = vertices_seq.shape
			vertices_tensor = torch.tensor(vertices_seq)
			faces_tensor = torch.tensor(np.tile(faces, (num_seq, 1, 1)))
			meshes = Meshes(verts=vertices_tensor, faces=faces_tensor)
			normals = meshes.verts_normals_packed()
			normals = normals.reshape(num_seq, -1, 3)

			normals_numpy = normals.cpu().numpy()
			renderable_mesh_seq = MeshSequence(vert_seq=vertices_seq, norm_seq=normals_numpy, faces=faces, color=self.colors[otype])

			self.mesh_seq_dict[otype] = renderable_mesh_seq


	def add_cur_camera_to_dict(self):
		cam_info = {'pos': self.cam_cur.pos, 'origin': self.cam_cur.origin}
		print(f"frame: {self.cur_frame} pos:{self.cam_cur.pos} origin: {self.cam_cur.origin}")
		self.camera_dict[self.cur_frame] = cam_info


	def keyboard_callback(self, key):
		if key == b"s":
			self.cur_time = 0.0
			self.cur_frame = 0
			self.time_checker.begin()
		elif key == b"]":
			self.cur_frame += 1
		elif key == b"[":
			self.cur_frame -= 1
		elif key == b"+":
			self.cur_frame += 5
		elif key == b"-":
			self.cur_frame -= 5
		elif key == b" ":
			self.play = not self.play

		# camera
		elif key == b"t":
			self.camera_mode = "tracking"
		elif key == b"f":
			self.camera_mode = "fixed"
		elif key == b"m":
			self.camera_mode = "free"
		elif key == b"e":
			self.read_cams = True
			self.cur_time = 0.0
			self.cur_frame = 0
			self.time_checker.begin()

		# more options
		elif key == b"o":
			self.pcd_point_size -= 0.25
			self.pcd_point_size = max(0.5, self.pcd_point_size)			
			glPointSize(self.pcd_point_size)
			print(f"point size to: {self.pcd_point_size}")

		elif key == b"p":
			self.pcd_point_size += 0.25
			glPointSize(self.pcd_point_size)
			print(f"point size to: {self.pcd_point_size}")

		elif key == b"c":
			# edit partially
			self.record_cams = True
			self.camera_dict = {}

		elif key == b"k":
			self.cur_time = 0.0
			self.cur_frame = 0
			self.time_checker.begin()
			self.record_cams = True
			self.camera_dict = {}

		# pcd show 
		elif key == b"i":
			self.show_pcd = not self.show_pcd


		# record
		elif key == b"r":
			self.cur_frame = 0
			self.time_checker.begin()
			
			if self.use_msaa:
				self._init_single_sample_fbo()
				self._init_msaa()

			print(f"Current pointcloud size: {self.pcd_point_size}")
			
			# save cam key
			save_name = input("Enter folder name to store screenshots: ")
			if self.record_cams:
				save_cam_path = os.path.join(path_constants.VIDEO_PATH, f"cam_keys/{save_name}.pkl")
				with open(save_cam_path, "wb") as file:
					pickle.dump(self.camera_dict, file)

			self.read_cams = True

			# save 
			save_path = os.path.join(path_constants.VIDEO_PATH, save_name)
			utils.create_dir_if_absent(save_path)

			num_frames = self.result_dict['motion']['result'].num_frames()

			start_end_frame_str = input("Enter frame range to record: ")
			start_end_frame = start_end_frame_str.split()
			start_frame = int(start_end_frame[0])
			end_frame = int(start_end_frame[1])
			if end_frame == -1:
				end_frame = num_frames-1

			# capture images
			img_count = 0
			for i in tqdm(range(start_frame, end_frame+1)):
				self.record = True
				self.cur_frame = i
				name = "output_%05d" % (img_count)
				self.save_screen(dir=save_path, name=name, render=True, save_alpha_channel=False)
				img_count += 1

			# run ffmpeg in 30fps
			input_path = f'{save_path}/output_%05d.png'
			output_video = f'/{save_path}/video.mp4'
			ffmpeg_command = [
				'ffmpeg',
				'-framerate', '30',
				'-i', input_path,
				'-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # This line is added for automatic adjustment
				'-c:v', 'libx264',
				'-pix_fmt', 'yuv420p',
				output_video
			]
			subprocess.run(ffmpeg_command, check=True)

			# save egocentric video
			proceed_save_egocentric = input("Write y if you want to save egocentric video: ")
			if proceed_save_egocentric == "y" and 'img_texture' in self.result_dict:
				images = self.result_dict['img_texture'][start_frame:end_frame+1]
				fps = 30  # Frames per second
				frame_size = (images[0].shape[1], images[0].shape[0])  # width, height of the first image
				output_file = f'/{save_path}/ego_video.mp4'

				fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for MP4 files
				video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
				for img in tqdm(images):
					video_writer.write(img)
				video_writer.release()

			self.record = False


		else:
			return False

		return True
	
	def add_light(self):
		glDisable(GL_CULL_FACE)
		glEnable(GL_DEPTH_TEST)

		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

		glEnable(GL_DITHER)
		glShadeModel(GL_SMOOTH)
		glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

		glDepthFunc(GL_LEQUAL)
		glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
	
		ambient = [0.065, 0.065, 0.065, 1.0]
		diffuse = [0.2, 0.2, 0.2, 1.0]

		# get bbox 
		bbox = self.result_dict['pcd'].get_axis_aligned_bounding_box()

		# Extracting the bbox coordinates
		xmin, ymin, zmin = bbox.get_min_bound()
		xmax, ymax, zmax = bbox.get_max_bound()

		# Positions for the 4 lights based on the bbox
		position1 = [xmin-1.0, ymin-1.0, zmax, 1.0]
		position2 = [xmin-1.0, ymax+1.0, zmax, 1.0]
		position3 = [xmax+1.0, ymin-1.0, zmax, 1.0]
		position4 = [xmax+1.0, ymax+1.0, zmax, 1.0]

		# Setting up the 4 lights
		glEnable(GL_LIGHT0)
		glLightfv(GL_LIGHT1, GL_AMBIENT, ambient)
		glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse)
		glLightfv(GL_LIGHT1, GL_POSITION, position1)

		glEnable(GL_LIGHT1)
		glLightfv(GL_LIGHT2, GL_AMBIENT, ambient)
		glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuse)
		glLightfv(GL_LIGHT2, GL_POSITION, position2)

		glEnable(GL_LIGHT2)
		glLightfv(GL_LIGHT3, GL_AMBIENT, ambient)
		glLightfv(GL_LIGHT3, GL_DIFFUSE, diffuse)
		glLightfv(GL_LIGHT3, GL_POSITION, position3)

		glEnable(GL_LIGHT3)
		glLightfv(GL_LIGHT4, GL_AMBIENT, ambient)
		glLightfv(GL_LIGHT4, GL_DIFFUSE, diffuse)
		glLightfv(GL_LIGHT4, GL_POSITION, position4)

		glEnable(GL_LIGHTING)
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_LIGHTING)

		print("Add light")
		

	def _render_pose(self, pose, color):
		skel = pose.skel
		for j in skel.joints:
			T = pose.get_transform(j, local=False)
			pos = conversions.T2p(T)
			joint_color =np.array([0.3,0.3,0.3])
			if self.render_motion_type == "gt":
				self.thickness = 0.02
			else:
				self.thickness = 0.015
			gl_render.render_point(pos, radius= self.thickness*1.5 * self.scale, color=joint_color)

			if j.parent_joint is not None:
				# returns X that X dot vec1 = vec2
				pos_parent = conversions.T2p(
					pose.get_transform(j.parent_joint, local=False)
				)
				p = 0.5 * (pos_parent + pos)
				l = np.linalg.norm(pos_parent - pos)
				r = self.thickness*1.2
				R = math.R_from_vectors(np.array([0, 0, 1]), pos_parent - pos)
				gl_render.render_capsule(
					conversions.Rp2T(R, p),
					l,
					r * self.scale,
					color=color,
					slice=8,
				)			
			

	def _render_characters(self):
		if self.camera_mode != "tracking":
			return 
		
		# output motion
		i = 0
		for otype in self.result_dict['motion']:
			motion = self.result_dict['motion'][otype]
			if motion is None:
				continue
			if self.render_motion_type != "all" and otype != self.render_motion_type:
				continue
			if otype == "gt":
				continue
			
			skel = motion.skel
			frame = self.cur_frame
			pose = motion.get_pose_by_frame(frame)
			if not self.read_cams and self.camera_mode == "tracking":
				self.set_tracking_camera(pose=pose)
				
			i += 1

	def _render_mesh_seq(self):
		for otype in self.result_dict['motion']:
			if otype == "gt":
				continue
			if self.cur_frame >= self.result_dict['motion'][otype].num_frames():
				cur_frame = self.result_dict['motion'][otype].num_frames()-1
			else:
				cur_frame = self.cur_frame
			self.mesh_seq_dict[otype].render_cur_mesh(cur_frame)


	def load_pcd_renderables(self):
		self.result_dict['pcd_render'] = PointCloud(pcd=self.result_dict['pcd'])


	def init_pcd_renderables(self):
		self.result_dict['pcd_render'].init_pcd()
		self.is_pcd_render_init = True
		glPointSize(self.pcd_point_size)
		glEnable(GL_POINT_SMOOTH)
	


	def generate_root_height_for_tracking_camera_filter(self):
		from scipy import ndimage

		for frame_idx in range(self.result_dict['motion']['result'].num_frames()):
			self.root_record.append(self.result_dict['motion']['result'].poses[frame_idx].get_root_transform()[2,3])
		
		# filter
		root_record_np = np.array(self.root_record)
		root_record_np = ndimage.uniform_filter1d(root_record_np, size=5, axis=0, mode="nearest")
		self.root_record = root_record_np

	def set_tracking_camera(self, pose):
		if len(self.root_record) == 0:
			self.generate_root_height_for_tracking_camera_filter()
		
		root_pos = pose.get_root_transform()[:3,3]
		root_pos[2] = self.root_record[self.cur_frame]
		delta = root_pos - self.cam_cur.origin
		self.cam_cur.origin += delta
		self.cam_cur.pos += delta 


	def set_fixed_camera(self):
		if len(list(self.camera_dict)) == 0:
			cam_pos = self.cam_cur.pos 
			cam_origin = self.cam_cur.origin
		else:
			cam_frame_keys = list(self.camera_dict)[-1]
			cam_pos = self.camera_dict[cam_frame_keys]['pos']
			cam_origin = self.camera_dict[cam_frame_keys]['origin']
		self.cam_cur.origin = cam_origin
		self.cam_cur.pos = cam_pos

	def _render_pcd(self):
		self.result_dict['pcd_render'].render_pcd()


	def _render_contact_height(self):
		contact_dict = self.result_dict['contact']
		# embed()
		# print("render contact")

		possible_floor_frame = []
		for cframe in contact_dict:
			if cframe < self.cur_frame:
				possible_floor_frame.append(cframe)

		floor = max(possible_floor_frame) if len(possible_floor_frame) > 0 else 0

		for cframe in contact_dict:
			if cframe >= self.cur_frame:
				continue
			root_pos = self.result_dict['motion']['result'].poses[cframe].get_transform(key="Hips", local=False)
			root_pos = conversions.T2p(root_pos)

			height = contact_dict[cframe]

			size = [2.4,2.4] if cframe == floor else [1.5,1.5]
			dsize = [0.8,0.8,0.8] if cframe == floor else [0.5, 0.5, 0.5]
			color = [0.3, 0.3, 0.7] if cframe == floor else [0.8, 0.2, 0.2]
			# fillin = True if cframe == floor else False

			glPushMatrix()
			glTranslatef(root_pos[0],root_pos[1],height)
			gl_render.render_ground(
				size=size,
				dsize=dsize,
				color=color,
				axis=motion_constants.UP_AXIS,
				origin=False,
				use_arrow=False,
				fillIn=False,
				line_width=3.0
			)
			glPopMatrix()
	

	def render_callback(self):
		
		# self._render_ground()

		if self.is_pcd_render_init is False:
			self.add_light()
			self.init_pcd_renderables()
			self.is_pcd_render_init = True
		if self.is_mesh_render_init is False:
			for otype in self.mesh_seq_dict:
				self.mesh_seq_dict[otype].init_mesh()

		self._render_mesh_seq()
		
		# self._render_characters() 
		if self.show_pcd:
			self._render_pcd()

		self._render_contact_height()

		if self.read_cams:
			self.read_cam_dict_and_set()
			return 
		
		if not self.read_cams and self.camera_mode == "tracking":
			pose = self.result_dict['motion']['result'].poses[self.cur_frame]
			self.set_tracking_camera(pose=pose)
		if self.record_cams:
			self.record_cam_key()


	def idle_callback(self):
		if self.camera_mode == "fixed":
			self.set_fixed_camera()
		time_elapsed = self.time_checker.get_time(restart=False)
		if time_elapsed < (1/30.0):
			offset_time = 1/30.0 - time_elapsed
			time.sleep(offset_time)
		if self.play:
			self.cur_time += self.play_speed * time_elapsed
			self.cur_frame += 1

		# self.cur_time += self.play_speed * time_elapsed
		self.time_checker.begin()

	def image_callback(self):
		if self.cur_frame >= len(self.result_dict['img_texture']):
			cur_frame = len(self.result_dict['img_texture']) - 1
		else:
			cur_frame = self.cur_frame
		self.data_texture = cv2.cvtColor(self.result_dict['img_texture'][cur_frame],cv2.COLOR_BGR2RGB)
		self.render_image()

	def _render_ground(self):
		glPushMatrix()
		glTranslatef(0,0,0.05)
		gl_render.render_ground(
			size=[100, 100],
			color=[0.6, 0.6, 0.6],
			axis=motion_constants.UP_AXIS,
			origin=False,
			use_arrow=False,
			fillIn=False
		)
		glPopMatrix()
		
	def render_image(self):
		
		glDisable(GL_CULL_FACE)
		glDisable(GL_DEPTH_TEST)
		glDisable(GL_LIGHTING)
		glEnable(GL_TEXTURE_2D)

		# glUseProgram(0)
		# embed()
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.data_texture.shape[1], self.data_texture.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, self.data_texture.data)
		texHeight,texWidth =   self.data_texture.shape[:2]
		texHeight *= 0.5
		texWidth *= 0.5

		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

		x_offset = 25
		y_offset = 25

		window_height = texHeight + y_offset + 50
		window_width = texWidth + x_offset + 50

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
		glBegin(GL_QUADS)
		glTexCoord2f(0.0, 0)
		glVertex2f(x_offset, y_offset)

		glTexCoord2f(1, 0)
		glVertex2f(window_width, y_offset)

		glTexCoord2f(1, 1)
		glVertex2f(window_width, window_height)

		glTexCoord2f(0, 1.0)
		glVertex2f(x_offset, window_height)        
		glEnd()

		glEnable(GL_LIGHTING)
		glEnable(GL_CULL_FACE)
		glEnable(GL_DEPTH_TEST)
		glDisable(GL_TEXTURE_2D)
	
	def overlay_callback(self):
		def format_three_decimals(x):
			return "{:.3f}".format(x)

		if self.record:
			return 
		w, h = self.window_size
		gl_render.render_text(
			f"Frame number: {self.cur_frame}",
			pos=[0.05 * w, 0.95 * h],
			font=GLUT_BITMAP_HELVETICA_18,
		)
		if 'img_texture' in self.result_dict:
			self.image_callback()
		return


def load_visualizer(result_dict, args):
	v_up_env = utils.str_to_axis(args.axis_up)    
	cam = camera.Camera(
		pos=np.array(args.camera_position),
		origin=np.array(args.camera_origin),
		vup=v_up_env,
		fov=45.0,
	)

	viewer = RealdataNetworkViewer(
		result_dict=result_dict,
		play_speed=args.speed,
		scale=args.scale,
		thickness=args.thickness,
		render_overlay=args.render_overlay,
		hide_origin=args.hide_origin,
		title="Viewer",
		cam=cam,
		size=(1920, 1080),
		# size=(2560, 1440),
	)
	
	return viewer
