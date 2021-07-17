import os
import torch
import torch.optim as optim
import torch.utils.data as data
import hydra
import numpy as np
from utils.realview_loader import RealviewScannetDataset, collate_image
from utils.virtualview_loader import VirtualviewScannetDataset
from modeling.deeplab import DeepLab
from unet import UNet
from visdom import Visdom
class Pipeline:
    """
    A pipeline for train & test 2D images
    """
    def __init__(self, cfg):
        self.cfg = cfg
    
    def train(self):   
        
        # get device (cuda or cpu)
        device = self.get_device()

        # first, get a collection of different scenes
        scene_dataset = self.get_scene_dataset()

        # model
        model = self.get_model().to(device)

        # optimizer
        optimizer = None
        lr = self.cfg.optimizer.learning_rate
        if self.cfg.optimizer.name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, 
                weight_decay=self.cfg.optimizer.weight_decay)
        
        # loss function
        loss_function = torch.nn.CrossEntropyLoss()

        model.train()

        # visualize curve
        if self.cfg.visdom.use:
            viz = Visdom(env=self.cfg.visdom.env)


        num_epoch = self.cfg.num_epoch
        for epoch in range(num_epoch): 

            for scene_id in range(len(scene_dataset)): # loop over scenes
                image_dataset = scene_dataset[scene_id]['color_imgset']
                image_dataloader = data.DataLoader(
                    dataset=image_dataset,
                    batch_size=self.cfg.data_loader.batch_size,
                    shuffle=True,
                    num_workers=self.cfg.data_loader.num_workers,
                    collate_fn=collate_image)

                for idx, batch in enumerate(image_dataloader): # loop over image
                    
                    # torch.cuda.empty_cache()

                    img = batch['img'].to(device)
                    semantic_label = batch['semantic_label'].to(device)
                    #instance_label = batch['instance_label'].to(device)

                    pred = model(img)
                    
                    loss = loss_function(pred, semantic_label)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if idx%10==0:
                        print('epoch %d idx %d loss:%f '%(epoch, idx, loss.item()))
                        if self.cfg.visdom.use:
                            viz.line(
                                X = np.array([idx]),
                                Y = np.array([loss.item()]),
                                win = 'epoch%d'%epoch,
                                opts= dict(title = 'epoch%d'%epoch),
                                update = 'append')

            self.save_model(model, self.cfg.model.model_name+'_epoch%d'%epoch)

    def evaluation(self):
        pass
    
    def save_model(self, model, model_name):
        save_path = os.path.join(self.cfg.model.save_model_path, model_name+'.pth')
        torch.save(model.state_dict(), save_path)


    def get_scene_dataset(self):
        if self.cfg.dataset.real_view: # use real view
            dataset = RealviewScannetDataset(self.cfg, mode='train')
        else: # use virtual view
            dataset = VirtualviewScannetDataset(self.cfg, mode='train')

        return dataset


    def get_model(self):
        # load model
        if self.cfg.model.model_name=='deeplabv3+':
            backbone = self.cfg.model.backbone
            num_classes = self.cfg.model.num_classes
            output_stride = self.cfg.model.output_stride
            model = DeepLab(backbone=backbone, num_classes=num_classes, 
                            output_stride=output_stride)
        elif self.cfg.model.model_name=='unet':
            num_channels = self.cfg.model.num_channels
            num_classes = self.cfg.model.num_classes
            model = UNet(n_channels=num_channels, n_classes=num_classes)

        # load pretrained model
        if self.cfg.model.pretrain:
            pretrain_path = self.cfg.model.pretrained_model_path
            model.load_state_dict(torch.load(pretrain_path))
        
        return model


    def get_device(self):
        device = None
        if self.cfg.device!='cpu':
            device = torch.device(self.cfg.device)
        else:
            device = torch.device('cpu')
        return device
            


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    ppl = Pipeline(cfg)
    ppl.train()
    
if __name__=='__main__':
    
    main()