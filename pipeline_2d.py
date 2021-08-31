import os
import torch
import torch.optim as optim
import torch.utils.data as data
import hydra
import numpy as np
from utils.realview_loader import RealviewScannetDataset, collate_image
from utils.virtualview_loader import VirtualviewScannetDataset
from utils.color_to_label import label_to_color

from visdom import Visdom
from utils.miou import miou_2d
from PIL import Image

class Pipeline2D:
    """
    A pipeline for training & evaluating 2D images
    """
    def __init__(self, cfg):
        self.cfg = cfg
        print('Use visdom:', cfg.visdom.use)
        print('Use virtual view:', not cfg.dataset.real_view)
    
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
        loss_function = torch.nn.CrossEntropyLoss(reduction='sum')

        model.train()

        # visualize curve
        if self.cfg.visdom.use:
            viz = Visdom(env=self.cfg.visdom.env)


        num_epoch = self.cfg.num_epoch
        for epoch in range(num_epoch): 

            for scene_id in range(1): # loop over scenes
                image_dataset = scene_dataset[scene_id]['imgset']
                image_dataloader = data.DataLoader(
                    dataset=image_dataset,
                    batch_size=self.cfg.data_loader.batch_size,
                    shuffle=False,
                    num_workers=self.cfg.data_loader.num_workers,
                    collate_fn=collate_image)

                for idx, batch in enumerate(image_dataloader): # loop over image
                    
                    # torch.cuda.empty_cache()

                    img = batch['color_img'].to(device)
                    semantic_label = batch['semantic_label'].to(device)
                    #instance_label = batch['instance_label'].to(device)
                  

                    pred = model(img)
                    pred = torch.softmax(pred,dim=1)
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
                            if idx%100==0:# do some eval
                                pred_label = torch.max(pred, dim=1).indices

                                # label_color_img = label_to_color(pred_label[0].cpu().numpy())

                                # label_color_img = Image.fromarray(label_color_img.astype(np.uint8))
                                # label_color_img.save('/home/yanjie/zyj_test/virtual-multi-view/train.png')
                                
                                # gt_img = img[0].permute(1,2,0)
                                # gt_img = Image.fromarray(gt_img.cpu().numpy().astype(np.uint8))
                                # gt_img.save('/home/yanjie/zyj_test/virtual-multi-view/gt.png')

                                mean_iou = miou_2d(pred_label.cpu(), semantic_label.cpu())
                                mean_iou = mean_iou.sum()/len(mean_iou)  # get an average of all imgs' miou
                                print('Evaluation: epoch %d idx %d miou:%f'%(epoch, idx, mean_iou))
                            print('epoch %d idx %d loss:%f'%(epoch, idx, loss.item()))
                    
                    
            
            is_save = True
            if epoch%10==0 and is_save:
                self.save_model(model, self.cfg.model.model_name+'_epoch%d'%epoch)
                print('Model in epoch%u saved. '%epoch)


    def evaluation(self):

        model_path = self.cfg.evaluation.model_path

        # get device (cuda or cpu)
        device = self.get_device()

        # first, get a collection of different scenes
        scene_dataset = self.get_scene_dataset(
            mode='train',
            use_transform=True
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
                with torch.no_grad():
                    img = batch['color_img'].to(device)
                    semantic_label = batch['semantic_label'].to(device)
                    #instance_label = batch['instance_label'].to(device)

                    pred = model(img)
                   
                    pred_label = torch.max(pred, dim=1).indices

                    mean_iou = miou_2d(pred_label.cpu(), semantic_label.cpu())
                    print(mean_iou)
                # if idx%10==0:
                #     if self.cfg.visdom.use:
                #         viz.line(
                #             X = np.array([idx]),
                #             Y = np.array([loss.item()]),
                #             win = 'scene%d'%(scene_id),
                #             opts= dict(title = 'scene%d'%(scene_id)),
                #             update = 'append')
                #     else:
                #         print('idx %d loss:%f '%(idx, loss.item()))


    
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
        
        print('Model name:', self.cfg.model.model_name)
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
    
    # mode = 'eval'
    mode = 'train'
    
    ppl = Pipeline2D(cfg)
    if mode == 'train':
        ppl.train()
    elif mode == 'eval':
        ppl.evaluation()
    
if __name__=='__main__':
    
    main()