try:
    from render.single_renderer import Renderer
except:
    from single_renderer import Renderer

import trimesh
import numpy as np
from plyfile import PlyData
import pyrender
import math
from utils.color_to_label import color_to_label


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
        self.available_poses = self.get_available_poses(self.pc_xyz)
    

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
    def get_available_poses(pc_xyz:np.array):
        """
        Return: four top corners of the point cloud
        """
        # TODO: check whether we need to divide 10
        x_min = np.min(pc_xyz[0])/10
        x_max = np.max(pc_xyz[0])/10
        y_min = np.min(pc_xyz[1])/10
        y_max = np.max(pc_xyz[1])/10
        z_min = np.min(pc_xyz[2])/10
        z_max = np.max(pc_xyz[2])/10

        available_poses = []
        # a pose is: x, y, z, roll, pitch, yaw
        available_poses.append([1., 1., 2.9, 60, 0, 320])
        
        # TODO: add other poses. Currently only one pose.

        return available_poses

    def render_one_image(self, pose, width, height):
        """
        Params: a pose matrix, width, height

        Return: a color img, a depth img
        """
        renderer = Renderer()
        renderer.add_mesh(self.load_3dmesh(self.mesh_file_path)) # Note that it's necessary to reload the mesh file
        color, depth = renderer.render_one_image(intrinsic_matrix=self.intrinsic, 
                                                pose_matrix=pose, 
                                                viewport_height=height, 
                                                viewport_width=width)
        return color, depth

    def render_one_groundtruth(self, pose, width, height):
        """
        Params: a pose matrix, width, height

        Return: color label img, label img
        """
        renderer = Renderer()
        gt_mesh_file_path = self.mesh_file_path.replace('.ply', '.labels.ply')
        renderer.add_mesh(self.load_3dmesh(gt_mesh_file_path)) # Note that it's necessary to reload the mesh file

        color_img, _ = renderer.render_one_image(intrinsic_matrix=self.intrinsic, 
                                                pose_matrix=pose, 
                                                viewport_height=height, 
                                                viewport_width=width,
                                                is_gt=True)

        label_img = color_to_label(color_img)
        return color_img, label_img

    def render_some_images(self, img_num:int=4, width:int=640, height:int=480):
        """
        Params: number of imgs to need
        
        Return: color_list, depth_list, pose_list, color_label_list, label_list
        """
        color_list = []
        depth_list = []
        pose_list = []
        color_label_list = []
        label_list = []
        x_min = self.pc_range[0]
        x_max = self.pc_range[1]
        y_min = self.pc_range[2]
        y_max = self.pc_range[3]
        z_min = self.pc_range[4]
        z_max = self.pc_range[5]
        

        for pose in self.available_poses:
           
            pose_matrix = self.get_camera_pose_matrix(x=pose[0],
                                                    y=pose[1],
                                                    z=pose[2],
                                                    roll=pose[3], # degree
                                                    pitch=pose[4],
                                                    yaw=pose[5])
            
            try: # if render fails, skip
                color_img, depth_img = self.render_one_image(pose=pose_matrix, width=width, height=height)
                color_label_img, label_img= self.render_one_groundtruth(pose=pose_matrix, width=width, height=height)

            except:
                print('This pose fail to be rendered (x,y,z,roll,pitch,yaw):\n', pose)
                continue
            color_list.append(color_img)
            depth_list.append(depth_img)
            pose_list.append(pose_matrix)
            color_label_list.append(color_label_img)
            label_list.append(label_img)

        return color_list, depth_list, pose_list, color_label_list, label_list
    

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
 