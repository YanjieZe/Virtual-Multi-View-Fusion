try:
    from render.single_renderer import Renderer
except:
    from single_renderer import Renderer

import trimesh
import numpy as np
from plyfile import PlyData
import pyrender

class MultiRenderer:
    def __init__(self, mesh_file_path):
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
        x_min = np.min(self.pc_xyz[0])
        x_max = np.max(self.pc_xyz[0])
        y_min = np.min(self.pc_xyz[1])
        y_max = np.max(self.pc_xyz[1])
        z_min = np.min(self.pc_xyz[2])
        z_max = np.max(self.pc_xyz[2])
        print(x_min,x_max)
        print(y_min,y_max)
        print(z_min,z_max)
        # TODO: 获得3D bounding box的表示方式，完善sample

    def load_3dmesh(self, mesh_file_path):
        fuze_trimesh = trimesh.load(mesh_file_path)
        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
        return mesh
    

    def render_one_image(self, pose):
        renderer = Renderer()
        renderer.add_mesh(self.mesh)
        color, depth = renderer.render_one_image(self.intrinsic, pose)
        return color, depth

    def render_some_images(self, img_num=10):
        color_list = []
        depth_list = []
        pose_list = []
        x0 = 0
        y0 = 0
        z0 = 0
        for i in range(img_num):#TODO: how to sample?
            self.get_camera_pose_matrix()

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
 