try:
    from render.single_renderer import Renderer
except:
    from single_renderer import Renderer

import trimesh
import numpy as np
from plyfile import PlyData
import pyrender
import math

class MultiRenderer:
    """
    Render multiple imgs at one time from one point cloud
    """
    def __init__(self, mesh_file_path):
        self.mesh_file_path = mesh_file_path
        self.mesh = self.load_3dmesh(mesh_file_path)
        self.intrinsic = None
        # get vertices
        with open(mesh_file_path, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
            vertices[:, 0] = plydata['vertex'].data['x']
            vertices[:, 1] = plydata['vertex'].data['y']
            vertices[:, 2] = plydata['vertex'].data['z']
            vertices[:, 3] = plydata['vertex'].data['red']
            vertices[:, 4] = plydata['vertex'].data['green']
            vertices[:, 5] = plydata['vertex'].data['blue']
        # not aligned
        self.pc_xyz = vertices[:3] # decimetre(dm)
        self.pc_rgb = vertices[3:]

        # range: x_min, x_max, y_min, y_max, z_min, z_max
        self.pc_range = self.get_3dminmax(self.pc_xyz)
        self.top_corners = self.get_top_corners(self.pc_xyz)
        
    @staticmethod
    def load_3dmesh(mesh_file_path:str):
        fuze_trimesh = trimesh.load(mesh_file_path)
        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
        return mesh

    @staticmethod
    def get_3dminmax(pc_xyz:np.array):
        """
        Return: x_min, x_max, y_min, y_max, z_min, z_max
        """
        x_min = np.min(pc_xyz[0])
        x_max = np.max(pc_xyz[0])
        y_min = np.min(pc_xyz[1])
        y_max = np.max(pc_xyz[1])
        z_min = np.min(pc_xyz[2])
        z_max = np.max(pc_xyz[2])
        return [x_min,x_max,y_min,y_max,z_min,z_max]

    @staticmethod
    def get_top_corners(pc_xyz:np.array):
        """
        Return: four top corners of the point cloud
        """
        x_min = np.min(pc_xyz[0])
        x_max = np.max(pc_xyz[0])
        y_min = np.min(pc_xyz[1])
        y_max = np.max(pc_xyz[1])
        z_min = np.min(pc_xyz[2])
        z_max = np.max(pc_xyz[2])

        corners = []
        corners.append(np.array([x_min,y_min,z_max]))
        corners.append(np.array([x_min,y_max,z_max]))
        corners.append(np.array([x_max,y_min,z_max]))
        corners.append(np.array([x_max,y_max,z_max]))

        return corners

    def render_one_image(self, pose):
        """
        Params: a pose matrix

        Return: a color img, a depth img
        """
        renderer = Renderer()
        renderer.add_mesh(self.load_3dmesh(self.mesh_file_path))
        color, depth = renderer.render_one_image(self.intrinsic, pose)
        return color, depth

    def render_some_images(self, img_num:int=4):
        """
        Params: number of imgs to need
        
        Return: list of Color img, list of Depth img, list of Pose
        """
        color_list = []
        depth_list = []
        pose_list = []

        x_min = self.pc_range[0]
        x_max = self.pc_range[1]
        y_min = self.pc_range[2]
        y_max = self.pc_range[3]
        z_min = self.pc_range[4]
        z_max = self.pc_range[5]
        
        # TODO: 修改拍摄位置和拍摄角度
        for corner_corordinate in self.top_corners:
            single_pose = self.get_camera_pose_matrix(x=corner_corordinate[0],
                                                    y=corner_corordinate[1],
                                                    z=corner_corordinate[2],
                                                    roll=40, # degree
                                                    pitch=0,
                                                    yaw=30)
            try: # if render fails, skip
                color_img, depth_img = self.render_one_image(single_pose)
            except:
                print('This pose fail to be rendered:\n', single_pose)
                print('Position is:',corner_corordinate)
                continue
            color_list.append(color_img)
            depth_list.append(depth_img)
            pose_list.append(single_pose)

        return color_list, depth_list, pose_list
    

    @staticmethod
    def get_camera_pose_matrix(x=0., y=0., z=0., roll=0., pitch=0., yaw=0.):
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


    def load_camera_intrinsic_matrix(self, fx, fy, cx, cy):
        IntrinsicMatrix = np.zeros((4,4))

        IntrinsicMatrix[0][0] = fx 
        IntrinsicMatrix[1][1] = fy
        IntrinsicMatrix[0][2] = cx
        IntrinsicMatrix[1][2] = cy

        IntrinsicMatrix[2][2] = 1.0
        IntrinsicMatrix[3][3] = 1.0

        self.intrinsic = IntrinsicMatrix

        return IntrinsicMatrix
 