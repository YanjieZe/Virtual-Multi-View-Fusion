from render.multi_renderer import MultiRenderer
from utils.virtualview_loader import VirtualviewScannetDataset
import hydra
import plyfile
from PIL import Image
import os
import numpy as np
import png as pypng

class ImgCreator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root_path = self.cfg.img_creator.save_path
        self.scene_dataset = VirtualviewScannetDataset(cfg, mode='train',use_transform=True)
        
    def fetch_one_scene(self, idx):
        return self.scene_dataset[idx]


    @staticmethod   
    def save_color_img(img:np.array, save_path:str):
        """
        Save color img(array) as img
        """
        image = Image.fromarray(img)
        image.save(save_path)


    @staticmethod    
    def save_pose(pose:np.array, save_path:str):
        """
        Save pose as txt
        """
        np.savetxt(save_path, pose)

    @staticmethod
    def save_depth_img(img:np.array, save_path:str):
        """
        Save a depth img
        """
        img_uint16 = np.round(img).astype(np.uint16)
        w_depth = pypng.Writer(img.shape[1], img.shape[0], greyscale=True, bitdepth=16)
        with open(save_path, 'wb') as f:
            w_depth.write(f, np.reshape(img_uint16, (-1, img.shape[1]))) 


    @staticmethod
    def save_gt_img(img:np.array, save_path:str):
        """
        Save a gt img
        """
        img_uint16 = np.round(img).astype(np.uint16)
        w_depth = pypng.Writer(img.shape[1], img.shape[0], greyscale=True, bitdepth=16)
        with open(save_path, 'wb') as f:
            w_depth.write(f, np.reshape(img_uint16, (-1, img.shape[1])))


    def create_imgs_from_one_scene(self, one_scene, is_save=True):
        """
        Params: one scene's data(directly taken from the dataset)
        """
        scene_name = one_scene['scene_name']
        mesh_file_path = one_scene['mesh_file_path']
        pc_xyz = one_scene['point_cloud_xyz']

        multi_renderer = MultiRenderer(mesh_file_path)
        intrinsic = multi_renderer.load_camera_intrinsic_matrix(fx=self.cfg.img_creator.fx, 
                                                                fy=self.cfg.img_creator.fy, 
                                                                cx=self.cfg.img_creator.cx, 
                                                                cy=self.cfg.img_creator.cy)
        color_list, depth_list, pose_list,gt_list = multi_renderer.render_some_images(
                                                img_num=self.cfg.img_creator.img_num,
                                                width=self.cfg.img_creator.width,
                                                height=self.cfg.img_creator.height)
        
        if is_save:
            # create dir
            scene_path = os.path.join(self.root_path,scene_name)
            color_path = os.path.join(scene_path, 'color')
            depth_path = os.path.join(scene_path, 'depth')
            pose_path = os.path.join(scene_path, 'pose')
            gt_path = os.path.join(scene_path, 'label')
            if not os.path.exists(scene_path):
                os.mkdir(scene_path)
            if not os.path.exists(color_path):
                os.mkdir(color_path)
            if not os.path.exists(depth_path):
                os.mkdir(depth_path)
            if not os.path.exists(pose_path):
                os.mkdir(pose_path)
            if not os.path.exists(gt_path):
                os.mkdir(gt_path)
            for i in range(len(color_list)):
                self.save_color_img(img=color_list[i], save_path=os.path.join(color_path,'%u.jpg'%i))
                self.save_gt_img(img=gt_list[i], save_path=os.path.join(gt_path, '%u.png'%i))
                self.save_depth_img(img=depth_list[i], save_path=os.path.join(depth_path,'%u.png'%i))
                self.save_pose(pose=pose_list[i], save_path=os.path.join(pose_path, '%u.txt'%i))

    
@hydra.main(config_path='config', config_name='config')
def main(cfg):
    img_creator = ImgCreator(cfg)
    idx = 0
    one_scene = img_creator.fetch_one_scene(idx=idx)
    img_creator.create_imgs_from_one_scene(one_scene)


if __name__=='__main__':
    main()

