import numpy as np
import trimesh
import pyrender
from pyrender import RenderFlags
import matplotlib.pyplot as plt
import os
import math
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['DISPLAY'] = ':0.0'
import warnings



class Renderer:
    """
    A Renderer that can load point cloud, compute pose matrix, and render imgs.
    """
    def __init__(self):
        self.scene = pyrender.Scene()

    def load_3dmesh(self, mesh_file):
        fuze_trimesh = trimesh.load(mesh_file)
        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
        self.scene.add(mesh)

    def add_mesh(self, mesh):
        self.scene.add(mesh)
        
    def render_one_image(self,
                        intrinsic_matrix, 
                        pose_matrix, 
                        z_near=0.01, 
                        z_far=100, 
                        light_color=[1.0, 1.0, 1.0],# RGB
                        light_intensity=5.0,
                        viewport_width=640,
                        viewport_height=400,
                        point_size=0.1):
        """
        return: RGB image, Depth image
        """
        
        # camera
        fx = intrinsic_matrix[0][0]
        fy = intrinsic_matrix[1][1]
        cx = intrinsic_matrix[0][2]
        cy = intrinsic_matrix[1][2]

        camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=z_near, zfar=z_far, name=None)
            
        self.scene.add(camera, pose=pose_matrix)

        # light
        # light = pyrender.SpotLight(color=np.ones(3)*255, intensity=100.0,
        #                             innerConeAngle=np.pi/16.0,
        #                             outerConeAngle=np.pi/6.0)
        # light = pyrender.PointLight(color=[255,255, 255], intensity=250.0) 
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_intensity)                           
        
        self.scene.add(light, pose=pose_matrix)                        

        # render image
        r = pyrender.OffscreenRenderer(viewport_width=viewport_width,
                                        viewport_height=viewport_height,
                                    point_size=point_size)
        flags =  RenderFlags.SHADOWS_DIRECTIONAL
        color, depth = r.render(self.scene, flags=flags)

        return color, depth
        

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
        
    
def demo():
    # ignore warnings 
    warnings.filterwarnings('ignore')
    
    # test_file = "scene0000_00_vh_clean_2.labels.ply"
    test_file = "../data/scene0000_00_vh_clean_2.ply"

    # create a renderer
    new_renderer = Renderer()

    # load file
    new_renderer.load_3dmesh(test_file)

    # camera params load
    # 50 0 30 n
    # 50 30 0 n
    # 0 50 30 n
    # 0 30 50 n
    # 30 50 0 
    # 30 0 50 
    pose_matrix = new_renderer.get_camera_pose_matrix(5.5, 1, 4, 40, 0, 30)

    ### intrinsic_matrix = new_renderer.get_camera_intrinsic_matrix()
    ### or
    intrinsic_matrix = np.array([
        [1169.621094 ,0.000000, 646.295044, 0.000000],
    [0.000000, 1167.105103, 489.927032, 0.000000],
    [0.000000 , 0.000000 ,1.000000 , 0.000000],
    [0.000000 ,0.000000 ,0.000000 ,1.000000],
    ])

    intrinsic_matrix = np.array([
        [300 ,0.000000, 640, 0.000000],
    [0.000000, 200, 480, 0.000000],
    [0.000000 , 0.000000 ,1.000000 , 0.000000],
    [0.000000 ,0.000000 ,0.000000 ,1.000000],
    ])

    # render
    color, depth = new_renderer.render_one_image(intrinsic_matrix, pose_matrix)

    # show the image
    plt.figure()
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(color)
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(depth)
    plt.savefig("1.png")

if __name__=='__main__':
    
    demo()
    