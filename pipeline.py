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
        scene_dataset = self.get_scene_dataset(
            mode='train',
            use_transform=True
        )

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
                image_dataset = scene_dataset[scene_id]['imgset']
                image_dataloader = data.DataLoader(
                    dataset=image_dataset,
                    batch_size=self.cfg.data_loader.batch_size,
                    shuffle=True,
                    num_workers=self.cfg.data_loader.num_workers,
                    collate_fn=collate_image)

                for idx, batch in enumerate(image_dataloader): # loop over image
                    
                    # torch.cuda.empty_cache()

                    img = batch['color_img'].to(device)
                    semantic_label = batch['semantic_label'].to(device)
                    #instance_label = batch['instance_label'].to(device)

                    pred = model(img)
                    
                    loss = loss_function(pred, semantic_label)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if idx%10==0:
                        if self.cfg.visdom.use:
                            viz.line(
                                X = np.array([idx]),
                                Y = np.array([loss.item()]),
                                win = 'epoch%d scene%d'%(epoch, scene_id),
                                opts= dict(title = 'epoch%d scene%d'%(epoch, scene_id)),
                                update = 'append')
                        else:
                            print('epoch %d idx %d loss:%f '%(epoch, idx, loss.item()))



            self.save_model(model, self.cfg.model.model_name+'_epoch%d'%epoch)


    def evaluation(self):

        model_path = self.cfg.evaluation.model_path

        # get device (cuda or cpu)
        device = self.get_device()

        # first, get a collection of different scenes
        scene_dataset = self.get_scene_dataset(
            mode='test',
            use_transform=False
        )

        # model
        
        model = self.get_model(model_path).to(device)

        model.eval()

        # visualize curve
        if self.cfg.visdom.use:
            viz = Visdom(env=self.cfg.visdom.env)


        # for scene_id in range(len(scene_dataset)): # loop over scenes
        for scene_id in range(1): # loop over scenes
            image_dataset = scene_dataset[scene_id]['imgset']
            image_dataloader = data.DataLoader(
                dataset=image_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.cfg.data_loader.num_workers,
                collate_fn=collate_image)

            for idx, batch in enumerate(image_dataloader): # loop over image

                img = batch['color_img'].to(device)
                semantic_label = batch['semantic_label'].to(device)
                #instance_label = batch['instance_label'].to(device)

                pred = model(img)
                
                # TODO: add IOU
                
                if idx%10==0:
                    if self.cfg.visdom.use:
                        viz.line(
                            X = np.array([idx]),
                            Y = np.array([loss.item()]),
                            win = 'epoch%d scene%d'%(epoch, scene_id),
                            opts= dict(title = 'epoch%d scene%d'%(epoch, scene_id)),
                            update = 'append')
                    else:
                        print('epoch %d idx %d loss:%f '%(epoch, idx, loss.item()))


    
    def save_model(self, model, model_name):
        save_path = os.path.join(self.cfg.model.save_model_path, model_name+'.pth')
        torch.save(model.state_dict(), save_path)


    def get_scene_dataset(self, mode='train', use_transform=True):
        if self.cfg.dataset.real_view: # use real view
            dataset = RealviewScannetDataset(self.cfg, mode=mode, use_transform=use_transform)
        else: # use virtual view
            dataset = VirtualviewScannetDataset(self.cfg, mode=mode, use_transform=use_transform)

        return dataset


    def get_model(self, model_path=None):
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
            


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    
    mode = 'eval'
    
    ppl = Pipeline(cfg)
    if mode == 'train':
        ppl.train()
    elif mode == 'eval':
        ppl.evaluation()
    
if __name__=='__main__':
    
    main()