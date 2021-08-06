from render.multi_renderer import MultiRenderer
from utils.virtualview_loader import VirtualviewScannetDataset
import hydra
import plyfile

class ImgCreator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root_path = self.cfg.img_creator.save_path
        self.scene_dataset = VirtualviewScannetDataset(cfg, mode='train',use_transform=True)
        
    def fetch_one_scene(self, idx):
        return self.scene_dataset[idx]

    def create_imgs_from_one_scene(self, one_scene):
        scene_name = one_scene['scene_name']
        mesh_file_path = one_scene['mesh_file_path']
        pc_xyz = one_scene['point_cloud_xyz']

        multi_renderer = MultiRenderer(mesh_file_path)
        multi_renderer.load_camera_intrinsic_matrix(fx=self.cfg.img_creator.fx, 
                                                    fy=self.cfg.img_creator.fy, 
                                                    cx=self.cfg.img_creator.cx, 
                                                    cy=self.cfg.img_creator.cy)
        
        
    
@hydra.main(config_path='config', config_name='config')
def main(cfg):
    img_creator = ImgCreator(cfg)
    idx = 0
    one_scene = img_creator.fetch_one_scene(idx=idx)
    img_creator.create_imgs_from_one_scene(one_scene)


if __name__=='__main__':
    main()

