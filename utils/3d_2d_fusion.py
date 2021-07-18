from PIL import Image
import numpy as np
import torch
import hydra

class Fusioner:
    """
    Get Mesh, then project them into 2D image
    """
    def __init__(self, point_cloud, intrinsic):
        self.pc = point_cloud
        self.feature_vectors = np.zeros(self.pc.shape[0])
        self.intrinsic = intrinsic
    

    def projection(self, depth_img, pose_matrix):
        """
        With depth check
        """
        pass

@hydra.main(config_path='../config', config_name='config')
def demo(cfg):
    from realview_loader import RealviewScannetDataset
    dataset = RealviewScannetDataset(cfg)

    scene_idx = 0
    pc = dataset[scene_idx]['mesh_vertices']
    intrinsic_depth = dataset[scene_idx]['intrinsic_depth']
    intrinsic_color = dataset[scene_idx]['intrinsic_color']
    imgset = dataset[scene_idx]['imgset']

    img_idx = 0
    depth_img = imgset[img_idx]['depth_img']
    pose_matrix = imgset[img_idx]['pose_matrix']
    semantic_label = imgset[img_idx]['semantic_label']
    print(semantic_label.shape)
    # fusioner = Fusioner()


if __name__=='__main__':
    demo()
    
