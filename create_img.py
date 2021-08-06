from render.multi_renderer import MultiRenderer
from utils.virtualview_loader import VirtualviewScannetDataset
import hydra

class ImgCreator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root_path = cfg.save_path
        self.scene_dataset = VirtualviewScannetDataset(cfg, mode='train',use_transform=True)
        
    def fetch_one_scene(self, idx):
        return self.scene_dataset[idx]

    def create_img_from_one_scene(self, one_scene):
        scene_name = one_scene['scene_name']
        mesh_file_path = one_scene['mesh_file_path']

        multi_renderer = MultiRenderer(mesh_file_path)
        

    
    

    
    
@hydra.main(config_path='config', config_name='config')
def main(cfg):
    img_creator = ImgCreator(cfg)
    idx = 0
    one_scene = img_creator.fetch_one_scene(idx=idx)
    create_img_from_one_scene(one_scene)



