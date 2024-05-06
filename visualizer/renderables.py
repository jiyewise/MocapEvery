import os
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from IPython import embed 
import sys, os
import numpy as np
from copy import deepcopy 
from fairmotion.utils import utils

class PointCloud:
    def __init__(self, pcd=None):
        self.g_vao= None
        self.g_vertex_buffer= None
        self.g_color_buffer=None        
        self.seq_len = 0
        self.batch_len = 0

        if pcd is not None:
            self.points = np.asarray(pcd.points)
            self.colors = np.asarray(pcd.colors)
            
            self.vts = self.points.astype(np.float32).flatten()
            self.vts_num = int(len(self.vts) / 3)

    def init_pcd(self):
        self.g_vao = glGenVertexArrays(1)
        self.g_vertex_buffer = glGenBuffers(1)
        self.g_color_buffer = glGenBuffers(1)

        # vao
        glBindVertexArray(self.g_vao)

        # vbo
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.g_vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, len(self.vts) * sizeof(ctypes.c_float), (ctypes.c_float * len(self.vts))(*self.vts), GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        # color attribute (hand idx is 47)
        colors = deepcopy(self.colors)
        colors = colors.astype(np.float32).flatten()

        glEnableVertexAttribArray(3)
        glBindBuffer(GL_ARRAY_BUFFER, self.g_color_buffer)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBufferData(GL_ARRAY_BUFFER, len(colors) * sizeof(ctypes.c_float), (ctypes.c_float * len(colors))(*colors), GL_DYNAMIC_DRAW)

        # unbind vao
        glBindVertexArray(0)    
        glPointSize(4)


    def render_pcd(self):
        # get positions and colors from frame idx
        # new_positions = self.vts_sequence[idx].astype(np.float32).flatten()

        # if new_colors is None:
        #     new_colors = np.ones((8192, 4)) * 0.4 
        #     new_colors[self.semantics[idx] == 47, :] = [1, 0, 0, 1]
        #     indexes = [i for i, value in enumerate(self.semantics[idx]) if value == 47]
        #     # print(indexes)
        #     # new_colors = new_colors[:,:3].astype(np.float32).flatten()

        # new_colors = new_colors[:,:3].astype(np.float32).flatten()
        
        # glBindBuffer(GL_ARRAY_BUFFER, self.g_vertex_buffer)
        # glBufferSubData(GL_ARRAY_BUFFER, 0, new_positions.nbytes, new_positions)

        # glBindBuffer(GL_ARRAY_BUFFER, self.g_color_buffer)
        # glBufferSubData(GL_ARRAY_BUFFER, 0, new_colors.nbytes, new_colors)

        # draw
        glBindVertexArray(self.g_vao)
        glDrawArrays(GL_POINTS, 0, self.vts_num)    
        glBindVertexArray(0)    # unbind vao


class Mesh:
    def __init__(self, mesh=None, color=np.array([0.6, 0.6, 0.6, 1])):
        self.mesh = mesh
        self.g_vao= None
        self.g_vertex_buffer= None
        self.g_normal_buffer= None
        # self.g_tangent_buffer = None
        self.g_index_buffer = None
        self.use_vertex_normal_color = False
        self.color = color

        if mesh is not None:
            self.load_from_trimesh(mesh=mesh)

    def load_from_trimesh(self, mesh):
        self.vts = mesh.vertices.flatten()
        self.inds = mesh.faces.flatten()
        self.normals = mesh.vertex_normals.flatten()
        self.tangent = self.normals
        self.vts_num = int(len(self.vts) / 3)
        self.face_num = int(len(self.inds) / 3)
        normal_num = len(self.normals) / 3
    

    def init_mesh(self):
        self.g_vao = glGenVertexArrays(1)
        self.g_vertex_buffer = glGenBuffers(1)
        self.g_tangent_buffer = glGenBuffers(1)
        self.g_normal_buffer = glGenBuffers(1)
        self.g_index_buffer = glGenBuffers(1)
        self.g_color_buffer = glGenBuffers(1)

        glBindVertexArray(self.g_vao)
        # vbo
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.g_vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, len(self.vts) * sizeof(ctypes.c_float), (ctypes.c_float * len(self.vts))(*self.vts), GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        if self.normals is not None:
            # tangent attribute
            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.g_tangent_buffer)
            glBufferData(GL_ARRAY_BUFFER, len(self.tangent) * sizeof(ctypes.c_float), (ctypes.c_float * len(self.tangent))(*self.tangent), GL_STATIC_DRAW)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

            # normal attribute
            glEnableVertexAttribArray(2)
            glBindBuffer(GL_ARRAY_BUFFER, self.g_normal_buffer)
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, None)
            glBufferData(GL_ARRAY_BUFFER, len(self.normals) * sizeof(ctypes.c_float), (ctypes.c_float * len(self.normals))(*self.normals), GL_STATIC_DRAW)

            if self.use_vertex_normal_color:
                # use normal value as color 
                # color attribute
                glEnableVertexAttribArray(3)
                glBindBuffer(GL_ARRAY_BUFFER, self.g_normal_buffer)
                glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, None)
                glBufferData(GL_ARRAY_BUFFER, len(self.normals) * sizeof(ctypes.c_float), (ctypes.c_float * len(self.normals))(*self.normals), GL_STATIC_DRAW)

            else:
                expanded_color = np.tile(self.color, (self.vts_num, 1)).flatten()  # shape: (10475, 4)
                glEnableVertexAttribArray(3)
                glBindBuffer(GL_ARRAY_BUFFER, self.g_color_buffer)
                glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, None)
                glBufferData(GL_ARRAY_BUFFER, len(expanded_color) * sizeof(ctypes.c_float), (ctypes.c_float * len(expanded_color))(*expanded_color), GL_STATIC_DRAW)
                
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.g_index_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ctypes.c_uint) * len(self.inds), (ctypes.c_uint * len(self.inds))(*self.inds), GL_STATIC_DRAW)
        glBindVertexArray(0)    # unbind vao


    def render_mesh(self):
        # draw
        glBindVertexArray(self.g_vao)    
        glDrawElements(GL_TRIANGLES, self.face_num * 3, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)    # Unbind


class MeshSequence:
    def __init__(self, vert_seq, norm_seq, faces, color):
        self.g_vao= None
        self.g_vertex_buffer= None
        self.g_normal_buffer= None
        # self.g_tangent_buffer = None
        self.g_index_buffer = None
        self.use_vertex_normal_color = False
        self.color = color

        # faces are fixed
        self.inds = faces.flatten()
        self.face_num = int(len(self.inds) / 3)

        self.vts_sequence = vert_seq
        self.normal_sequence = norm_seq
        
        self.vts = self.vts_sequence[0].astype(np.float32).flatten()
        self.vts_num = int(len(self.vts) / 3)
        self.inds = faces.flatten()
        self.normals = self.normal_sequence[0].flatten()


    # def load_vert_seq(self, vert_seq, faces):

        # self.num_points = pc_sequence.num_points 
        # self.seq_len = pc_sequence.seq_len
        # self.batch_len = pc_sequence.batch_len 

        # self.vts = self.vts_sequence[0].astype(np.float32).flatten()
        # self.vts_num = int(len(self.vts) / 3)
        # self.colors = pc_sequence.color # initial color

        # self.inds = trimesh_seq[]

        # self.vts_seq = vert_seq
        # # self.normal_seq = compute_normals_batched(vertex_sequences=vert_seq, faces=faces)
        
        # self.vts = self.vts_seq[0].astype(np.float32).flatten()
        # self.vts_num = int(len(self.vts) / 3)
        # self.inds = faces.flatten()
        # self.face_num = int(len(self.inds) / 3)
        # # self.normals = self.normal_seq[0].flatten()
        # self.normals = None
        

    def init_mesh(self):
        self.g_vao = glGenVertexArrays(1)
        self.g_vertex_buffer = glGenBuffers(1)
        self.g_tangent_buffer = glGenBuffers(1)
        self.g_normal_buffer = glGenBuffers(1)
        self.g_index_buffer = glGenBuffers(1)
        self.g_color_buffer = glGenBuffers(1)

        glBindVertexArray(self.g_vao)
        # vbo
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.g_vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, len(self.vts) * sizeof(ctypes.c_float), (ctypes.c_float * len(self.vts))(*self.vts), GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        if self.normals is not None:
            # tangent attribute
            # glEnableVertexAttribArray(1)
            # glBindBuffer(GL_ARRAY_BUFFER, self.g_tangent_buffer)
            # glBufferData(GL_ARRAY_BUFFER, len(self.tangent) * sizeof(ctypes.c_float), (ctypes.c_float * len(self.tangent))(*self.tangent), GL_DYNAMIC_DRAW)
            # glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

            # normal attribute
            glEnableVertexAttribArray(2)
            glBindBuffer(GL_ARRAY_BUFFER, self.g_normal_buffer)
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, None)
            glBufferData(GL_ARRAY_BUFFER, len(self.normals) * sizeof(ctypes.c_float), (ctypes.c_float * len(self.normals))(*self.normals), GL_DYNAMIC_DRAW)

            if self.use_vertex_normal_color:
                # use normal value as color 
                # color attribute
                glEnableVertexAttribArray(3)
                glBindBuffer(GL_ARRAY_BUFFER, self.g_normal_buffer)
                glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, None)
                glBufferData(GL_ARRAY_BUFFER, len(self.normals) * sizeof(ctypes.c_float), (ctypes.c_float * len(self.normals))(*self.normals), GL_DYNAMIC_DRAW)

            else:
                color = self.color
                expanded_color = np.tile(color, (self.vts_num, 1)).flatten()  # shape: (10475, 4)
                glEnableVertexAttribArray(3)
                glBindBuffer(GL_ARRAY_BUFFER, self.g_color_buffer)
                glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, None)
                glBufferData(GL_ARRAY_BUFFER, len(expanded_color) * sizeof(ctypes.c_float), (ctypes.c_float * len(expanded_color))(*expanded_color), GL_DYNAMIC_DRAW)
                
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.g_index_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ctypes.c_uint) * len(self.inds), (ctypes.c_uint * len(self.inds))(*self.inds), GL_DYNAMIC_DRAW)
        glBindVertexArray(0)    # unbind vao


    # def render_mesh(self):
    #     # draw
    #     glBindVertexArray(self.g_vao)    
    #     glDrawElements(GL_TRIANGLES, self.face_num * 3, GL_UNSIGNED_INT, None)
    #     glBindVertexArray(0)    # Unbind


    def render_cur_mesh(self, idx):
        # get positions and colors from frame idx
        new_positions = self.vts_sequence[idx].astype(np.float32).flatten()
        new_normals = self.normal_sequence[idx].astype(np.float32).flatten()

        glBindBuffer(GL_ARRAY_BUFFER, self.g_vertex_buffer)
        glBufferSubData(GL_ARRAY_BUFFER, 0, new_positions.nbytes, new_positions)

        glBindBuffer(GL_ARRAY_BUFFER, self.g_normal_buffer)
        glBufferSubData(GL_ARRAY_BUFFER, 0, new_normals.nbytes, new_normals)

        # glBindBuffer(GL_ARRAY_BUFFER, self.g_color_buffer)
        # glBufferSubData(GL_ARRAY_BUFFER, 0, new_colors.nbytes, new_colors)

        # draw
        glBindVertexArray(self.g_vao)    
        glDrawElements(GL_TRIANGLES, self.face_num * 3, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)    # Unbind


# deprecated (pointcloud sequence)
class PointCloudSeq:
    def __init__(self, pc_seq=None):
        self.g_vao= None
        self.g_vertex_buffer= None
        self.g_color_buffer=None        
        self.seq_len = 0
        self.batch_len = 0
        if pc_seq is not None:
            self.from_pc_seq(pc_seq)

    def from_pc_seq(self, pc_sequence):

        self.vts_sequence = pc_sequence.vts_sequence
        
        # with h5py.File("./train2.h5", 'r') as f:
        #     self.vts_sequence = f['pcd'][0]*0.01
        #     self.semantics = np.array(f['semantic'][0]) # (300, 8192)
        #     self.batch_len, self.seq_len, self.num_points, _ = f['pcd'].shape # [300, 8192, 3]
        
        self.semantics = pc_sequence.semantics
        self.num_points = pc_sequence.num_points 
        self.seq_len = pc_sequence.seq_len
        self.batch_len = pc_sequence.batch_len 

        self.vts = self.vts_sequence[0].astype(np.float32).flatten()
        self.vts_num = int(len(self.vts) / 3)
        self.colors = pc_sequence.color # initial color

    def init_pcd(self):
        self.g_vao = glGenVertexArrays(1)
        self.g_vertex_buffer = glGenBuffers(1)
        self.g_color_buffer = glGenBuffers(1)

        # vao
        glBindVertexArray(self.g_vao)

        # vbo
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.g_vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, len(self.vts) * sizeof(ctypes.c_float), (ctypes.c_float * len(self.vts))(*self.vts), GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        # color attribute (hand idx is 47)
        colors = deepcopy(self.colors)
        colors = colors.astype(np.float32).flatten()

        glEnableVertexAttribArray(3)
        glBindBuffer(GL_ARRAY_BUFFER, self.g_color_buffer)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBufferData(GL_ARRAY_BUFFER, len(colors) * sizeof(ctypes.c_float), (ctypes.c_float * len(colors))(*colors), GL_DYNAMIC_DRAW)

        # unbind vao
        glBindVertexArray(0)    
        glPointSize(4)

    def render_pcd(self, idx, new_colors=None):
        # get positions and colors from frame idx
        new_positions = self.vts_sequence[idx].astype(np.float32).flatten()

        if new_colors is None:
            new_colors = np.ones((8192, 4)) * 0.4 
            new_colors[self.semantics[idx] == 47, :] = [1, 0, 0, 1]
            indexes = [i for i, value in enumerate(self.semantics[idx]) if value == 47]
            # print(indexes)
            # new_colors = new_colors[:,:3].astype(np.float32).flatten()

        new_colors = new_colors[:,:3].astype(np.float32).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, self.g_vertex_buffer)
        glBufferSubData(GL_ARRAY_BUFFER, 0, new_positions.nbytes, new_positions)

        glBindBuffer(GL_ARRAY_BUFFER, self.g_color_buffer)
        glBufferSubData(GL_ARRAY_BUFFER, 0, new_colors.nbytes, new_colors)

        # draw
        glBindVertexArray(self.g_vao)
        glDrawArrays(GL_POINTS, 0, self.num_points)    
        glBindVertexArray(0)    # unbind vao