# Copyright (c) Facebook, Inc. and its affiliates.
import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
# from OpenGL.raw.GL.ARB.vertex_array_object import glGenVertexArrays, \
                                      #   glBindVertexArray
import sys
import numpy as np
from fairmotion.viz import camera, utils

from PIL import Image

import trimesh
from IPython import embed 


# class Mesh:
#     def __init__(self, mesh):
#         self.mesh = mesh
#         self.g_vao= None
#         self.g_vertex_buffer= None
#         self.g_normal_buffer= None
#         # self.g_tangent_buffer = None
#         self.g_index_buffer = None

#         self.vts = mesh.vertices.flatten()
#         self.inds = mesh.faces.flatten()
#         self.normals = mesh.vertex_normals.flatten()
#         self.tangent = self.normals
#         self.vts_num = int(len(self.vts) / 3)
#         self.face_num = int(len(self.inds) / 3)
#         normal_num = len(self.normals) / 3
        
#         self.use_vertex_normal_color = False

#     def init_mesh(self):
#         self.g_vao = glGenVertexArrays(1)
#         self.g_vertex_buffer = glGenBuffers(1)
#         self.g_tangent_buffer = glGenBuffers(1)
#         self.g_normal_buffer = glGenBuffers(1)
#         self.g_index_buffer = glGenBuffers(1)
#         self.g_color_buffer = glGenBuffers(1)

#         glBindVertexArray(self.g_vao)
#         # vbo
#         glEnableVertexAttribArray(0)
#         glBindBuffer(GL_ARRAY_BUFFER, self.g_vertex_buffer)
#         glBufferData(GL_ARRAY_BUFFER, len(self.vts) * sizeof(ctypes.c_float), (ctypes.c_float * len(self.vts))(*self.vts), GL_STATIC_DRAW)
#         glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

#         if self.normals is not None:
#             # tangent attribute
#             # glEnableVertexAttribArray(1)
#             # glBindBuffer(GL_ARRAY_BUFFER, self.g_tangent_buffer)
#             # glBufferData(GL_ARRAY_BUFFER, len(self.tangent) * sizeof(ctypes.c_float), (ctypes.c_float * len(self.tangent))(*self.tangent), GL_STATIC_DRAW)
#             # glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

#             # normal attribute
#             glEnableVertexAttribArray(2)
#             glBindBuffer(GL_ARRAY_BUFFER, self.g_normal_buffer)
#             glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, None)
#             glBufferData(GL_ARRAY_BUFFER, len(self.normals) * sizeof(ctypes.c_float), (ctypes.c_float * len(self.normals))(*self.normals), GL_STATIC_DRAW)

#             if self.use_vertex_normal_color:
#                 # use normal value as color 
#                 # color attribute
#                 glEnableVertexAttribArray(3)
#                 glBindBuffer(GL_ARRAY_BUFFER, self.g_normal_buffer)
#                 glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, None)
#                 glBufferData(GL_ARRAY_BUFFER, len(self.normals) * sizeof(ctypes.c_float), (ctypes.c_float * len(self.normals))(*self.normals), GL_STATIC_DRAW)

#             else:
#                 color = np.array([0.6, 0.6, 0.6, 0.6])  # shape: (4,)
#                 expanded_color = np.tile(color, (self.vts_num, 1)).flatten()  # shape: (10475, 4)
#                 glEnableVertexAttribArray(3)
#                 glBindBuffer(GL_ARRAY_BUFFER, self.g_color_buffer)
#                 glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, None)
#                 glBufferData(GL_ARRAY_BUFFER, len(expanded_color) * sizeof(ctypes.c_float), (ctypes.c_float * len(expanded_color))(*expanded_color), GL_STATIC_DRAW)
                

#         glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.g_index_buffer)
#         glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ctypes.c_uint) * len(self.inds), (ctypes.c_uint * len(self.inds))(*self.inds), GL_STATIC_DRAW)
#         glBindVertexArray(0)    # unbind vao


#     def render_mesh(self):
#         # draw
#         glBindVertexArray(self.g_vao)    
#         glDrawElements(GL_TRIANGLES, self.face_num * 3, GL_UNSIGNED_INT, None)
#         glBindVertexArray(0)    # Unbind



class Viewer:
    """Viewer class builds general infrastructure to implement visualizer
    class for motion sequences.

    Attributes:
        title: Title displayed on visualizer window
        cam: Camera object for the scene
        size: Tuple; Visualizer window dimensions
        mouse_last_pos: Tuple; Stores last recorded position of mouse on the
            screen
        pressed_button: str; Stores last pressed keyboard character
        time_checker: Object of utils.TimeChecker class to keep track of UNIX
            time and playback time

    To create a custom visualizer, extend this class and implement the
    following methods:
        - render_callback
        - idle_callback
        - keyboard_callback (optional)
        - overlay_callback (optional)

    Once the viewer is initialized, call `run` method to display visualization
    """

    def __init__(
        self, title="glutgui_base", cam=None, size=(800, 600),
        bgcolor=[1.0, 1.0, 1.0, 1.0], use_msaa=True, mouse_sensitivity=0.005,
    ):
        self.title = title
        self.window = None
        self.window_size = size
        self.mouse_last_pos = None
        self.pressed_button = None
        self.bgcolor = bgcolor
        self.use_msaa = use_msaa
        self.mouse_sensitivity = mouse_sensitivity

        self.time_checker = utils.TimeChecker()
        if cam is None:
            self.cam_cur = camera.Camera(
                pos=np.array([0.0, 2.0, 4.0]),
                origin=np.array([0.0, 0.0, 0.0]),
                vup=np.array([0.0, 1.0, 0.0]),
                fov=45.0,
            )
        else:
            self.cam_cur = cam


    # def set_meshes(self, meshes=None):
    #     self.meshes = []
    #     if meshes is None:
    #         return
    #     for mesh in meshes:
    #         self.meshes.append(Mesh(mesh))

    def idle_callback(self):
        pass

    def overlay_callback(self):
        pass

    def keyboard_callback(self, key):
        return True

    def render_callback(self):
        return

    def _init_GL(self, w, h):
        glDisable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_DITHER)
        glShadeModel(GL_SMOOTH)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

        glDepthFunc(GL_LEQUAL)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

        glClearColor(
            self.bgcolor[0], 
            self.bgcolor[1], 
            self.bgcolor[2], 
            self.bgcolor[3])
        glClear(GL_COLOR_BUFFER_BIT)

        ambient = [0.2, 0.2, 0.2, 1.0]
        diffuse = [0.6, 0.6, 0.6, 1.0]
        front_mat_shininess = [60.0]
        front_mat_specular = [0.2, 0.2, 0.2, 1.0]
        front_mat_diffuse = [0.5, 0.28, 0.38, 1.0]
        lmodel_ambient = [0.2, 0.2, 0.2, 1.0]
        lmodel_twoside = [GL_FALSE]

        position = [1.0, 0.0, 0.0, 0.0]
        position1 = [-1.0, 1.0, 1.0, 0.0]
        position2 = [-1.0, -1.0, 1.0, 0.0]
        position3 = [-1.0, -1.0, -1.0, 0.0]


        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT0, GL_POSITION, position)

        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient)
        glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside)

        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT1, GL_POSITION, position1)

        glDisable(GL_LIGHTING)

        glEnable(GL_COLOR_MATERIAL)

        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, front_mat_specular)
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, front_mat_diffuse)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glDisable(GL_CULL_FACE)
        glEnable(GL_NORMALIZE)

        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
    
        if self.use_msaa:
            self._init_single_sample_fbo()
            self._init_msaa()
        
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)


    def resize_GL(self, w, h):
        self.window_size = (w, h)
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.cam_cur.fov, float(w) / float(h), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def draw_GL(self, swap_buffer=True):

        if self.use_msaa:
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.msaa_fbo)
        
        # Clear The Screen And The Depth Buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            *self.cam_cur.pos, *self.cam_cur.origin, *self.cam_cur.vup,
        )

        self.render_callback()

        if self.overlay_callback is not None:
            glClear(GL_DEPTH_BUFFER_BIT)
            glPushAttrib(GL_DEPTH_TEST)
            glDisable(GL_DEPTH_TEST)
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(
                0.0, self.window_size[0], self.window_size[1], 0.0, 0.0, 1.0
            )

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            self.overlay_callback()
            glPopMatrix()

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glPopAttrib()

        if self.use_msaa:
            self._post_process_msaa()

        if swap_buffer:
            glutSwapBuffers()

    def _init_msaa(self):
        num_samples = glGetIntegerv(GL_MAX_SAMPLES)

        print('num_samples_for_msaa:', num_samples)

        self.msaa_tex = glGenTextures(1)
        print('msaa_tex:', self.msaa_tex)
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, self.msaa_tex)
        glTexImage2DMultisample(
            GL_TEXTURE_2D_MULTISAMPLE, num_samples, GL_RGBA8, self.window_size[0], self.window_size[1], False)

        self.msaa_rbo_color = glGenRenderbuffers(1)
        print('msaa_rbo_color:', self.msaa_rbo_color)
        glBindRenderbuffer(GL_RENDERBUFFER, self.msaa_rbo_color)
        glRenderbufferStorageMultisample(
            GL_RENDERBUFFER, num_samples, GL_RGBA8, self.window_size[0], self.window_size[1])

        self.msaa_rbo_depth = glGenRenderbuffers(1)
        print('msaa_rbo_depth:', self.msaa_rbo_depth)
        glBindRenderbuffer(GL_RENDERBUFFER, self.msaa_rbo_depth)
        glRenderbufferStorageMultisample(
            GL_RENDERBUFFER, num_samples, GL_DEPTH_COMPONENT, self.window_size[0], self.window_size[1])

        self.msaa_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.msaa_fbo)

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, self.msaa_fbo, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self.msaa_rbo_color)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.msaa_rbo_depth)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        print("frame_buffer_status:", status)

        glBindFramebuffer(GL_FRAMEBUFFER, self.msaa_fbo)

        iMultiSample = glGetIntegerv(GL_SAMPLE_BUFFERS)
        iNumSamples = glGetIntegerv(GL_SAMPLES)
        print("MSAA on, GL_SAMPLE_BUFFERS = %d, GL_SAMPLES = %d\n"%(iMultiSample, iNumSamples))

    def _post_process_msaa(self):
        x, y, w, h = glGetIntegerv(GL_VIEWPORT)

        # Bind the multisampled FBO for reading
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.msaa_fbo)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glDrawBuffer(GL_BACK)
        glBlitFramebuffer(
            x, y, w, h, x, y, w, h, 
            GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, 
            GL_NEAREST)

    # The function called whenever a key is pressed.
    # Note the use of Python tuples to pass in: (key, x, y)
    def key_pressed(self, *args):
        handled = self.keyboard_callback(args[0])
        if handled:
            return
        if args[0] == b"\x1b":
            print("Hit ESC key to quit.")
            glutDestroyWindow(self.window)
            sys.exit()

    def mouse_func(self, button, state, x, y):
        if state == 0:
            self.pressed_button = button
        else:
            self.pressed_button = None

        if state == 0:  # Mouse pressed
            self.mouse_last_pos = np.array([x, y])
        elif state == 1:
            self.mouse_last_pos = None

        if button == 3:
            self.cam_cur.zoom(0.95)
        elif button == 4:
            self.cam_cur.zoom(1.05)

    def motion_func(self, x, y):
        newPos = np.array([x, y])
        d = self.mouse_sensitivity * (newPos - self.mouse_last_pos)
        if self.pressed_button == 0:
            self.cam_cur.rotate(d[1], -d[0], 0)
        elif self.pressed_button == 2:
            self.cam_cur.translate(np.array([d[0], d[1], 0]), frame_local=True)
        self.mouse_last_pos = newPos

    def render_timer(self, timer):
        glutPostRedisplay()
        glutTimerFunc(10, self.render_timer, 1)

    def run(self):
        # Init glut
        glutInit(())
        glutInitDisplayMode(
            GLUT_RGBA
            | GLUT_DOUBLE
            | GLUT_ALPHA
            | GLUT_DEPTH
            # | GLUT_MULTISAMPLE
        )
        glutInitWindowSize(*self.window_size)
        glutInitWindowPosition(0, 0)

        self.window = glutCreateWindow(self.title)

        # Init functions
        # glutFullScreen()
        glutDisplayFunc(self.draw_GL)
        glutIdleFunc(self.idle_callback)
        glutReshapeFunc(self.resize_GL)
        glutKeyboardFunc(self.key_pressed)
        glutMouseFunc(self.mouse_func)
        glutMotionFunc(self.motion_func)
        glutTimerFunc(10, self.render_timer, 1)
        self._init_GL(*self.window_size)
        self.time_checker.begin()

        # Run
        glutMainLoop()

    def save_screen(self, dir, name, format="png", render=False, save_alpha_channel=False):
        image = self.get_screen(render, save_alpha_channel)
        image.save(os.path.join(dir, "%s.%s" % (name, format)), format=format)

    # def get_screen(self, render=False, save_alpha_channel=False):
    #     if render:
    #         self.draw_GL()
        
    #     x, y, width, height = glGetIntegerv(GL_VIEWPORT)
    #     glPixelStorei(GL_PACK_ALIGNMENT, 1)
    #     if save_alpha_channel:
    #         data = glReadPixels(x, y, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    #         image = Image.frombytes("RGBA", (width, height), data)
    #     else:
    #         data = glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    #         image = Image.frombytes("RGB", (width, height), data)
    #     image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
    #     return image

    def _init_single_sample_fbo(self):
        # Create a single-sample texture and FBO
        self.single_sample_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.single_sample_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.window_size[0], self.window_size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

        self.single_sample_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.single_sample_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.single_sample_tex, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _blit_to_single_sample_fbo(self):
        # Bind the multisampled FBO for reading and single-sample FBO for drawing
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.msaa_fbo)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.single_sample_fbo)
        glBlitFramebuffer(0, 0, self.window_size[0], self.window_size[1], 0, 0, self.window_size[0], self.window_size[1], GL_COLOR_BUFFER_BIT, GL_NEAREST)

    def get_screen(self, render=False, save_alpha_channel=False):
        if render:
            self.draw_GL()

        # Blit the multisampled FBO to the single-sample FBO
        self._blit_to_single_sample_fbo()

        # Bind the single-sample FBO and read pixels from it
        glBindFramebuffer(GL_FRAMEBUFFER, self.single_sample_fbo)
        x, y, width, height = glGetIntegerv(GL_VIEWPORT)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        if save_alpha_channel:
            data = glReadPixels(x, y, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
            image = Image.frombytes("RGBA", (width, height), data)
        else:
            data = glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            image = Image.frombytes("RGB", (width, height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # Don't forget to reset FBO binding if necessary
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        return image

    
