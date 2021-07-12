import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['DISPLAY'] = ':0.0'

test_file = "data/scene0000_00_vh_clean_2.labels.ply"
fuze_trimesh = trimesh.load(test_file)

# mesh
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)

# scene
scene = pyrender.Scene()
scene.add(mesh)

# camera
camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.0)
s = np.sqrt(2)/2
camera_pose = np.array([
    [0.0, -s, s, 0.3],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, s, s, 0.35],
    [0.0, 0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)

# light
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                            innerConeAngle=np.pi/16.0,
                            outerConeAngle=np.pi/6.0)
scene.add(light, pose=camera_pose)                        

r = pyrender.OffscreenRenderer(viewport_width=640,
                                viewport_height=480,
                               point_size=1.0)
color, depth = r.render(scene)

# show the image
plt.figure()
plt.subplot(1,2,1)
plt.axis('off')
plt.imshow(color)
plt.subplot(1,2,2)
plt.axis('off')
plt.imshow(depth, cmap=plt.cm.gray_r)
plt.savefig("render.png")