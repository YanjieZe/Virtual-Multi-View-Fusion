from render.multi_renderer import MultiRenderer
from utils.virtualview_loader import VirtualviewScannetDataset
import hydra

class ImgCreator:
    def __init__(self, cfg):
        self.cfg = cfg
    
    # TODO: create imgs.
    
    
@hydra.main(config_path='config', config_name='config_imgcreator')
def main(cfg):
    img_creator = ImgCreator(cfg)


