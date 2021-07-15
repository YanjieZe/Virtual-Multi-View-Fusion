import torch
import torch.utils.data as data
import hydra
from utils.realview_loader import RealviewScannetDataset
from utils.virtualview_loader import VirtualviewScannetDataset
from modeling.deeplab import DeepLab

class Pipeline:
    """
    Yet another pipeline
    """
    def __init__(self, cfg):
        self.cfg = cfg
    
    def train(self):
        dataset = self.get_dataset()
        model = self.get_model()
        

    def get_dataset(self):
        if self.cfg.dataset.real_view: # use real view
            dataset = RealviewScannetDataset(self.cfg)
        else: # use virtual view
            dataset = VirtualviewScannetDataset(self.cfg)

        return dataset


    def get_model(self):
        backbone = self.cfg.model.backbone
        num_classes = self.cfg.model.num_classes
        output_stride = self.cfg.model.output_stride
        model = DeepLab(backbone=backbone, num_classes=num_classes, 
                        output_stride=output_stride)
        if self.cfg.model.pretrain:
            pretrain_path = self.cfg.model.pretrained_model_path
            model.load_state_dict(torch.load(pretrain_path))
        return model


    
    

@hydra.main(config_path="config", config_name="config")
def main(cfg):
    ppl = Pipeline(cfg)
    ppl.train()
    
if __name__=='__main__':
    main()