from PIL import Image
import numpy as np
import torch
import hydra

class Fusioner:
    """
    Get Mesh, then project them into 2D image
    """
    def __init__(self, point_cloud, intrinsic_depth, intrinsic_color):
        self.pc = point_cloud.unsqueeze(2)
        self.feature_vectors = np.zeros(self.pc.shape[0])
        self.intrinsic_depth = intrinsic_depth
        self.intrinsic_color = intrinsic_color
    

    def projection(self, depth_img, color_img, pose_matrix,semantic_label):
        """
        With depth check
        """
        # TODO: Is this extrinsic true?
        extrinsic = torch.inverse(pose_matrix)
        rotation_matrix = extrinsic[:3,:3] # 3*3
        translation_vector = extrinsic[:3,3].unsqueeze(1) # 3*1
        translation_vector = (torch.ones(self.pc.shape[0])*translation_vector).T.unsqueeze(2)# num_point*3*1
        
        project_points = torch.matmul(rotation_matrix, self.pc) + translation_vector # num_point*3*1
        
        # TODO: 下面写的转化不太对？
        intrinsic_color = self.intrinsic_color[:3,:3]
        intrinsic_depth = self.intrinsic_depth[:3,:3]
        
        project_points_color = torch.matmul(intrinsic_color, project_points)
        project_points_depth = torch.matmul(intrinsic_depth, project_points)
        print(project_points_depth[0])
        print(project_points_color[0])
        #pc = torch.mm(extrinsic, self.pc)
        
        

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
    color_img = imgset[img_idx]['color_img']
    pose_matrix = imgset[img_idx]['pose_matrix']
    semantic_label = imgset[img_idx]['semantic_label']
    
    fusion_machine = Fusioner(point_cloud=pc, 
                        intrinsic_depth=intrinsic_depth,
                        intrinsic_color=intrinsic_color)
    fusion_machine.projection(depth_img=depth_img, 
                            color_img=color_img,
                            pose_matrix=pose_matrix,
                            semantic_label=semantic_label
                            )

    # fusioner = Fusioner()



if __name__=='__main__':
    demo()
    
