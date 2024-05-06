
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

class MocapViewer(glut_viewer.Viewer):

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
        self.play_speed = play_speed
        self.render_overlay = render_overlay
        self.hide_origin = hide_origin
        self.motion_idx = 0
        self.cur_time = 0.0
        self.scale = scale
        self.thickness = thickness
        self.play = True

        # length of motion
        self.motion_num = len(self.result_dict['motion'])
        self.render_contact = False
        super().__init__(**kwargs)

    def keyboard_callback(self, key):
        motion = self.result_dict['motion'][self.motion_idx][0]
        if key == b"s":
            self.cur_time = 0.0
            self.time_checker.begin()
        if key == b"c":
            self.render_contact = not self.render_contact
        elif key == b"]":
            next_frame = min(
                motion.num_frames() - 1,
                motion.time_to_frame(self.cur_time) + 1,
            )
            self.cur_time = motion.frame_to_time(next_frame)
        elif key == b"[":
            prev_frame = max(0, motion.time_to_frame(self.cur_time) - 1)
            self.cur_time = motion.frame_to_time(prev_frame)
        elif key == b" ":
            self.play = not self.play
        elif key == b"q":
            if self.motion_idx > 0:
                self.motion_idx -= 1
            self.cur_time = 0.0
            self.idx = 0
            self.time_checker.begin()
        elif key == b"w":
            if self.motion_idx < self.motion_num-1:
                self.motion_idx += 1        
            self.cur_time = 0.0
            self.idx = 0
            self.time_checker.begin()
        elif (key == b"r" or key == b"v"):
            self.cur_time = 0.0
            end_time = motion.length()
            fps = motion.fps
            save_path = input(
                "Enter directory/file to store screenshots/video: "
            )
            cnt_screenshot = 0
            dt = 1 / fps
            gif_images = []
            while self.cur_time <= end_time:
                print(
                    f"Recording progress: {self.cur_time:.2f}s/{end_time:.2f}s ({int(100*self.cur_time/end_time)}%) \r",
                    end="",
                )
                if key == b"r":
                    utils.create_dir_if_absent(save_path)
                    name = "screenshot_%04d" % (cnt_screenshot)
                    self.save_screen(dir=save_path, name=name, render=True)
                else:
                    image = self.get_screen(render=True)
                    gif_images.append(
                        image.convert("P", palette=Image.ADAPTIVE)
                    )
                self.cur_time += dt
                cnt_screenshot += 1
            if key == b"v":
                utils.create_dir_if_absent(os.path.dirname(save_path))
                gif_images[0].save(
                    save_path,
                    save_all=True,
                    optimize=False,
                    append_images=gif_images[1:],
                    loop=0,
                )
        else:
            return False

        return True
    
    def _render_pose(self, pose, color, is_output=False):
        skel = pose.skel
        for j in skel.joints:
            T = pose.get_transform(j, local=False)
            pos = conversions.T2p(T)
            joint_color =np.array([0.3,0.3,0.3])
            gl_render.render_point(pos, radius= self.thickness*1.2 * self.scale, color=joint_color)
            # if not is_output:
            #     if j.name in ["Head", "LeftHand", "RightHand"]:
            #         gl_render.render_point(pos, radius= self.thickness*2 * self.scale, color=np.array([0,0,0]))
            #         gl_render.render_transform(T, scale=0.05, use_arrow=True, line_width=3)
            #         T_translate_for_output = deepcopy(T)
            #         T_translate_for_output[:3,3] = T_translate_for_output[:3,3] + self.result_dict['translate_offset']
            #         gl_render.render_point(pos+self.result_dict['translate_offset'], radius= self.thickness*2 * self.scale, color=np.array([0,0,0]))
            #         gl_render.render_transform(T_translate_for_output, scale=0.05, use_arrow=True, line_width=3)

            if j.parent_joint is not None:
                # returns X that X dot vec1 = vec2
                pos_parent = conversions.T2p(
                    pose.get_transform(j.parent_joint, local=False)
                )
                p = 0.5 * (pos_parent + pos)
                l = np.linalg.norm(pos_parent - pos)
                r = self.thickness
                R = math.R_from_vectors(np.array([0, 0, 1]), pos_parent - pos)
                gl_render.render_capsule(
                    conversions.Rp2T(R, p),
                    l,
                    r * self.scale,
                    color=color,
                    slice=8,
                )
        
        # render contact
        if self.render_contact:
            frame_idx = self.get_frame_idx()
            for dir in self.result_dict['contacts']:
                contact = self.result_dict['contacts'][dir][frame_idx]
                if contact:
                    foot_joint_name = f"{dir}Foot"
                    gl_render.render_point(conversions.T2p(pose.get_transform(foot_joint_name, local=False)), radius=0.02, color=np.array([1,0,0]))            


    def _render_characters(self, colors):

        ref_and_pred = self.result_dict['motion'][self.motion_idx]
        for i, motion in enumerate(ref_and_pred):
            t = self.cur_time % motion.length()
            frame_idx = motion.time_to_frame(t)
            skel = motion.skel
            pose = motion.get_pose_by_frame(frame_idx)
            color = colors[i % len(colors)]
            self._render_pose(pose, color, is_output=bool(i))
    

    # def _render_foot_est(self):
    #     frame_idx = self.get_frame_idx()
    #     for est_key in self.result_dict['est']:

    #     foot_est = self.result_dict['foot_est'][self.motion_idx][frame_idx]
    #     for i_foot in foot_est:
    #         gl_render.render_sphere(conversions.p2T(i_foot), color=np.array([1,0,0]), r=0.02)


    def set_tracking_camera(self, pose):
        root_pos = pose.get_root_transform()[:3,3]
        root_pos[1] = 0.8
        delta = root_pos - self.cam_cur.origin
        self.cam_cur.origin += delta
        self.cam_cur.pos += delta 


    def render_callback(self):
        gl_render.render_ground(
            size=[100, 100],
            color=[0.6, 0.6, 0.6],
            axis=motion_constants.UP_AXIS,
            origin=not self.hide_origin,
            use_arrow=True,
            fillIn=False
        )
        colors = [
            np.array([123, 174, 85, 255]) / 255.0,  # green
            # np.array([255, 255, 0, 255]) / 255.0,  # yellow
            np.array([238, 150, 70, 255]) / 255.0,  # orange
            np.array([85, 160, 173, 255]) / 255.0,  # blue
        ]
        self._render_characters(colors)
        # self._render_input(frame_idx=frame_idx)
        # self._render_foot_est()

    def idle_callback(self):
        time_elapsed = self.time_checker.get_time(restart=False)
        if self.play:
            self.cur_time += self.play_speed * time_elapsed
        self.time_checker.begin()

    def get_frame_idx(self):
        seq_time_length = self.result_dict['seq_len'] / self.result_dict['fps']
        seq_time = self.cur_time % seq_time_length
        return int(seq_time * self.result_dict['fps'] + 1e-05)
    
    def overlay_callback(self):
        # render 
        current_motion_idx = self.result_dict['idx'][self.motion_idx]
        cur_frame_idx = self.get_frame_idx()
        w, h = self.window_size
        gl_render.render_text(
            f"Motion idx: {current_motion_idx} Frame number: {cur_frame_idx}",
            pos=[0.05 * w, 0.95 * h],
            font=GLUT_BITMAP_TIMES_ROMAN_24,
        )

        if False:
            w, h = self.window_size
            t = self.cur_time % self.motions[0][0].length()
            frame = self.motions[0][0].time_to_frame(t)
            gl_render.render_text(
                f"Frame number: {frame}",
                pos=[0.05 * w, 0.95 * h],
                font=GLUT_BITMAP_TIMES_ROMAN_24,
            )

            if len(self.filenames) > 0:
                gl_render.render_text(
                    f"Index: {self.filenames[self.file_idx]}",
                    pos=[0.05 * w, 0.15 * h],
                    font=GLUT_BITMAP_TIMES_ROMAN_24,
                )


def load_visualizer(result_dict, args):
    v_up_env = utils.str_to_axis(args.axis_up)    

    cam = camera.Camera(
        pos=np.array(args.camera_position),
        origin=np.array(args.camera_origin),
        vup=v_up_env,
        fov=45.0,
    )

    viewer = MocapViewer(
        result_dict=result_dict,
        play_speed=args.speed,
        scale=args.scale,
        thickness=args.thickness,
        render_overlay=args.render_overlay,
        hide_origin=args.hide_origin,
        title="IMU2Body Testset Viewer",
        cam=cam,
        size=(1280, 720),
    )
    
    return viewer

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Visualize BVH file with block body"
#     )
#     parser.add_argument("--result-dir", type=str, required=False)
#     parser.add_argument("--result-idx", type=int, default=0)
#     parser.add_argument("--translate", type=float, default=0.5)

#     from render_args import add_render_argparse
#     parser = add_render_argparse(parser=parser)
#     args = parser.parse_args()

#     # main(args)
