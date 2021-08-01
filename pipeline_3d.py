import torch
import numpy as np
import hydra
import torch.utils.data as data
import os
from utils.realview_loader import RealviewScannetDataset, collate_image
from utils.fusion_2d_3d import Fusioner
from utils.miou import miou_2d, miou_3d
from tqdm import tqdm

class Pipeline3D:
    """
    A pipeline supports 3D inference.
    """
    def __init__(self, cfg):
        self.cfg = cfg


    def inference(self):
        device = self.get_device()

        model_path = self.cfg.evaluation.model_path
        model = self.get_model(model_path).to(device)

        scene_dataset = self.get_scene_dataset(
            mode='train',
            use_transform=True)

        # for scene_id in range(len(scene_dataset)):
        for scene_id in range(1):
            single_scene_data = scene_dataset[scene_id]

            image_dataset = single_scene_data['imgset']
            pc_xyz = single_scene_data['point_cloud_xyz'].to(device)
            pc_semantic_label = single_scene_data['semantic_label'].to(device)
            intrinsic_depth = single_scene_data['intrinsic_depth'].to(device)

            # img loader
            image_dataloader = data.DataLoader(
                dataset=image_dataset,
                batch_size=self.cfg.data_loader.batch_size,
                shuffle=False,
                num_workers=self.cfg.data_loader.num_workers,
                collate_fn=collate_image)

            # a machine able to fuse 2d and 3d
            fusion_machine = Fusioner(point_cloud=pc_xyz,
                                intrinsic_depth=intrinsic_depth,
                                collection_method='average',
                                use_gt=False)

            # collect features
            process_bar = tqdm(total=len(image_dataset)) # show process
            batch_size = self.cfg.data_loader.batch_size

            for idx, batch in enumerate(image_dataloader): # loop over image
                with torch.no_grad():
                    imgs = batch['color_img'].to(device)
                    depth_imgs = batch['depth_img']
                    pose_matrixs = batch['pose_matrix']
                    semantic_labels = batch['semantic_label'].to(device)
                    
                    preds = model(imgs)
                
                    for i in range(imgs.shape[0]): #single data point
                        depth_img = depth_imgs[i].to(device)
                        img = imgs[i].to(device)
                        pose_matrix = pose_matrixs[i].to(device)
                        pred = preds[i].to(device)
                        
                        fusion_machine.projection(depth_img=depth_img,
                                                pose_matrix=pose_matrix,
                                                feature_img=pred,
                                                threshold=5.0)
                
                process_bar.update(batch_size)
                
            pc_features = fusion_machine.get_features()
            pc_pred_label = torch.max(pc_features, dim=1).indices
            # get MIOU
            MIOU = miou_3d(pc_pred_label, pc_semantic_label)

            print(MIOU)

    def get_model(self, model_path=None):
        # load model
        if self.cfg.model.model_name=='deeplabv3+':
            from modeling.deeplab import DeepLab
            backbone = self.cfg.model.backbone
            num_classes = self.cfg.model.num_classes
            output_stride = self.cfg.model.output_stride
            model = DeepLab(backbone=backbone, num_classes=num_classes, 
                            output_stride=output_stride)
        elif self.cfg.model.model_name=='unet':
            from unet import UNet
            num_channels = self.cfg.model.num_channels
            num_classes = self.cfg.model.num_classes
            model = UNet(n_channels=num_channels, n_classes=num_classes)

        # load pretrained model
        if model_path is None:
            if self.cfg.model.pretrain:
                pretrain_path = self.cfg.model.pretrained_model_path
                model.load_state_dict(torch.load(pretrain_path))
            
        # load self path
        if not model_path is None:
            model.load_state_dict(torch.load(model_path))
        

        return model


    def get_device(self):
        device = None
        if self.cfg.device!='cpu':
            device = torch.device(self.cfg.device)
        else:
            device = torch.device('cpu')
        return device
    
    def get_scene_dataset(self, mode='train', use_transform=True):
        if self.cfg.dataset.real_view: # use real view
            dataset = RealviewScannetDataset(self.cfg, mode=mode, use_transform=use_transform)
        else: # use virtual view
            dataset = VirtualviewScannetDataset(self.cfg, mode=mode, use_transform=use_transform)

        return dataset

@hydra.main(config_path='config', config_name='config')
def main(cfg):
    ppl = Pipeline3D(cfg)
    ppl.inference()

if __name__=='__main__':
    main()
    