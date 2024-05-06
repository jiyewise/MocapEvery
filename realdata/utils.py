import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
# import matplotlib
# matplotlib.use('Agg')
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from IPython import embed
from fairmotion.utils import utils 
from fairmotion.ops import conversions
from imu2body.functions import * 
import subprocess
import shutil
import constants.path as path_constants

def time_to_frame(timecode_list, timecode):
	differences = np.abs(timecode_list - timecode)
	index = np.argmin(differences)
	return index

def time_range_into_frames(timecode_list, start_timecode, end_timecode=None):
	if end_timecode is None:
		end_timecode = timecode_list[-1] # assuming ascending order

	return np.where((timecode_list >= start_timecode) & (timecode_list <= end_timecode))[0]    

# droid related

def parse_video(dir):
    video_path = os.path.join(path_constants.DATA_PATH, f"{dir}/video/ego_video.MP4")
    image_dir =  os.path.join(path_constants.DATA_PATH, f"{dir}/sfm/images/")
    image_path = image_dir + "output_%06d.png"
    utils.create_dir_if_absent(image_dir)

    command = f"ffmpeg -i {video_path} -vf fps=30 {image_path}"
    subprocess.run(command, shell=True, check=True)


def undistorted_img_to_video(dir):
    image_dir =  os.path.join(path_constants.DATA_PATH, f"{dir}/sfm/undistorted_images/")
    video_dir = os.path.join(path_constants.DATA_PATH, f"{dir}/sfm/undistorted_img_video/")
    utils.create_dir_if_absent(video_dir)

    ffmpeg_command = f"ffmpeg -framerate 30 -i {image_dir}undistort_%05d.png -c:v libx264 -crf 20 -pix_fmt yuv420p {video_dir}output.mp4"
    
    subprocess.run(ffmpeg_command, shell=True, check=True)


# reproject 
def reproject(cam_ext, cam_int, points,):
	cam_ext_inv = invert_T(cam_ext)
	points_in_cam_coord = (np.einsum('ij,kj->ki',conversions.T2R(cam_ext_inv), points) + conversions.T2p(cam_ext_inv))
	projected_points = np.einsum('ij,kj->ki',cam_int,points_in_cam_coord)
	projected_points /= projected_points[...,2][..., np.newaxis]	

	return projected_points 


def plot_scalars(scalars):
    plt.plot(scalars)
    plt.axhline(y=3, color='r', linestyle='--')
    plt.title("Plot of Camera Translation Velocity")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
    
def plot_3_overlaid(data1, data2=None, title=['','']):
	x1, y1, z1 = data1[:,0], data1[:,1], data1[:,2]

	# Create subplots
	fig, axs = plt.subplots(3, sharex=True, sharey=True)

	# Plot x values
	axs[0].plot(x1, color='r', label=title[0])
	axs[0].set_title('X values')
	axs[0].legend()

	# Plot y values
	axs[1].plot(y1, color='g', label=title[0])
	axs[1].set_title('Y values')
	axs[1].legend()

	# Plot z values
	axs[2].plot(z1, color='b', label=title[0])
	axs[2].set_title('Z values')
	axs[2].legend()

	if data2 is not None:    
		x2, y2, z2 = data2[:, 0], data2[:, 1], data2[:, 2]

		axs[0].plot(x2, color='r', linestyle='dashed', label=title[1])
		axs[1].plot(y2, color='g', linestyle='dashed', label=title[1])
		axs[2].plot(z2, color='b', linestyle='dashed', label=title[1])

	# Display the plots
	plt.tight_layout()
	plt.show()


def plot_3_2side(data1, data2, data1_2=None, data2_2=None, x_vline_list=None):
	x1, y1, z1 = data1[:,0], data1[:,1], data1[:,2]
	x2, y2, z2 = data2[:,0], data2[:,1], data2[:,2]

	# Create subplots
	fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)

	# x
	axs[0][0].plot(x1, color='r', label='watch (Left Hand)')
	axs[0][0].set_title('X values (Left Hand)')
	axs[0][0].legend()

	axs[0][1].plot(x2, color='r', label='watch (Right Hand)')
	axs[0][1].set_title('X values (Right Hand)')
	axs[0][1].legend()

	# y values for left and right hand
	axs[1][0].plot(y1, color='g', label='watch (Left Hand)')
	axs[1][0].set_title('Y values (Left Hand)')
	axs[1][0].legend()

	axs[1][1].plot(y2, color='g', label='watch (Right Hand)')
	axs[1][1].set_title('Y values (Right Hand)')
	axs[1][1].legend()

	# z values for left and right hand
	axs[2][0].plot(z1, color='b', label='watch (Left Hand)')
	axs[2][0].set_title('Z values (Left Hand)')
	axs[2][0].legend()

	axs[2][1].plot(z2, color='b', label='watch (Right Hand)')
	axs[2][1].set_title('Z values (Right Hand)')
	axs[2][1].legend()

	if data1_2 is not None:
		x1_2, y1_2, z1_2 = data1_2[:,0], data1_2[:,1], data1_2[:,2]
		axs[0][0].plot(x1_2, color='r', linestyle='dashed', label='syn (Left Hand)')
		axs[1][0].plot(y1_2, color='g', linestyle='dashed', label='syn (Left Hand)')
		axs[2][0].plot(z1_2, color='b', linestyle='dashed', label='syn (Left Hand)')

	if data2_2 is not None:
		x2_2, y2_2, z2_2 = data2_2[:,0], data2_2[:,1], data2_2[:,2]
		axs[0][1].plot(x2_2, color='r', linestyle='dashed', label='syn (Right Hand)')
		axs[1][1].plot(y2_2, color='g', linestyle='dashed', label='syn (Right Hand)')
		axs[2][1].plot(z2_2, color='b', linestyle='dashed', label='syn (Right Hand)')

	if x_vline_list is not None:
		for x_vline in x_vline_list:
			axs[0][0].axvline(x=x_vline, color='k', linestyle='--')
			axs[1][0].axvline(x=x_vline, color='k', linestyle='--')
			axs[2][0].axvline(x=x_vline, color='k', linestyle='--')

			# For Right Hand X, Y, Z plots
			axs[0][1].axvline(x=x_vline, color='k', linestyle='--')
			axs[1][1].axvline(x=x_vline, color='k', linestyle='--')
			axs[2][1].axvline(x=x_vline, color='k', linestyle='--')

	# Display the plots
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
    parse_video("1105")
