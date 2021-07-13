import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import os
import math
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['DISPLAY'] = ':0.0'

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




if __name__=='__main__':
    # read *.ply file
    test_file = "scene0000_00_vh_clean.ply"
    test_file = "data/scene0000_00_vh_clean_2.labels.ply"

    fuze_trimesh = trimesh.load(test_file)

    # mesh
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)

    # scene
    scene = pyrender.Scene()
    scene.add(mesh)

    # camera
    IntrinsicMatrix = np.array([
        [1169.621094 ,0.000000, 646.295044, 0.000000],
    [0.000000, 1167.105103, 489.927032, 0.000000],
    [0.000000 , 0.000000 ,1.000000 , 0.000000],
    [0.000000 ,0.000000 ,0.000000 ,1.000000],
    ])
    fx = IntrinsicMatrix[0][0]
    fy = IntrinsicMatrix[1][1]
    cx = IntrinsicMatrix[0][2]
    cy = IntrinsicMatrix[1][2]

    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.5, zfar=100, name=None)
    s = np.sqrt(2)/2
    # camera_pose = np.array([
    #     [-0.955421, 0.119616, -0.269932, 2.655830],
    # [0.295248, 0.388339, -0.872939 ,2.981598],
    # [0.000408, -0.913720 ,-0.406343, 1.368648],
    # [0.000000 ,0.000000 ,0.000000, 1.000000],
    # ])

    # camera_pose = np.array([
    #     [0.951595 ,0.120375 ,-0.282803, 1.655525],
    # [0.307283 ,0.392547 ,-0.866882, 1.981353],
    # [0.006663 ,-0.911820 ,-0.410535 ,0.261859],
    # [0.000000 ,0.000000, 0.000000 ,1.000000],
    # ]
    # )
    # camera_pose = np.array([
    #     [-0.984788 ,-0.061134 ,0.162653 ,2.760859],
    # [-0.173195 ,0.269766 ,-0.947222, 3.073889],
    # [0.014029 ,-0.960983, -0.276251 ,3.482407],
    # [0.000000, 0.000000 ,0.000000 ,1.000000],
    # ])

    camera_pose = get_camera_pose_matrix(2.7, 3.07, 3.48, 0, -30, 0)

    scene.add(camera, pose=camera_pose)

    # light
    # light = pyrender.SpotLight(color=np.ones(3)*255, intensity=100.0,
    #                             innerConeAngle=np.pi/16.0,
    #                             outerConeAngle=np.pi/6.0)
    # light = pyrender.PointLight(color=[255,255, 255], intensity=250.0) 
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)                           
    scene.add(light, pose=camera_pose)                        

    r = pyrender.OffscreenRenderer(viewport_width=640,
                                    viewport_height=400,
                                point_size=0.1)
    color, depth = r.render(scene)


    # show the image
    plt.figure()
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(color)
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(depth)
    plt.savefig("render.png")