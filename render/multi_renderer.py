try:
    from render.single_renderer import Renderer
except:
    from single_renderer import Renderer

import trimesh
import numpy as np


class MultiRenderer:
    def __init__(self, mesh_file_path):
        self.mesh = load_3dmesh(mesh_file_path)

    def load_3dmesh(self, mesh_file_path):
        fuze_trimesh = trimesh.load(mesh_file_path)
        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
        return mesh

    def render_one_image(self, pose):
        renderer = Renderer()
        renderer.add_mesh(self.mesh)
        # TODO:首先看一下axis align后的东西是啥，然后再渲染。

    def get_camera_pose_matrix(self, x=0., y=0., z=0., roll=0., pitch=0., yaw=0.):
        """
        Get Pose Matrix by six params: x, y, z, roll, pitch, yaw
        """
        translation_vector = np.array([x, y, z, 1]).reshape(4,1)
        R_roll = np.array([
            [1, 0, 0],
            [0, math.cos(math.pi/180*roll), -math.sin(math.pi/180*roll)],
            [0, math.sin(math.pi/180*roll), math.cos(math.pi/180*roll)],
        ])
        R_pitch = np.array([
            [math.cos(math.pi/180*pitch), 0, math.sin(math.pi/180*pitch)],
            [0, 1, 0],
            [-math.sin(math.pi/180*pitch),0, math.cos(math.pi/180*pitch)],
        ])
        R_yaw = np.array([
            [math.cos(math.pi/180*yaw), -math.sin(math.pi/180*yaw), 0],
            [math.sin(math.pi/180*yaw), math.cos(math.pi/180*yaw), 0],
            [0, 0, 1],
        ])
        R = np.matmul(R_yaw, R_pitch, R_roll)
        
        pose_matrix = np.vstack([R,np.zeros((1,3))])
        pose_matrix = np.hstack([pose_matrix, translation_vector])

        return pose_matrix


    def get_camera_intrinsic_matrix(self, fx, fy, cx, cy):
        IntrinsicMatrix = np.zeros((4,4))

        IntrinsicMatrix[0][0] = fx 
        IntrinsicMatrix[1][1] = fy
        IntrinsicMatrix[0][2] = cx
        IntrinsicMatrix[1][2] = cy

        IntrinsicMatrix[2][2] = 1.0
        IntrinsicMatrix[3][3] = 1.0

        return IntrinsicMatrix
 