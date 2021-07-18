import os
import hydra
import zipfile
@hydra.main(config_path='config', config_name='config')
def unzip(cfg):
    """
    Unzip some 2D zip file provided by ScanNet
    """
    train_path = cfg.dataset.train_path
    test_path = cfg.dataset.test_path
    path_list = [train_path, test_path]
    for root_path in path_list:
        scene_list = os.listdir(root_path)
        for scene_id in scene_list:

            label_path = os.path.join(root_path, scene_id, 'label-filt')
            if not os.path.exists(label_path):
                file_tounzip = os.path.join(root_path, scene_id, '%s_2d-label-filt.zip'%(scene_id))
                with zipfile.ZipFile(file_tounzip, 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(root_path, scene_id))
                print('finish %s'%scene_id)

def test():
    file_tozip = '/data/ScanNetV2/scans/scene0444_00/scene0444_00_2d-label-filt.zip'
    with zipfile.ZipFile(file_tozip, 'r') as zip_ref:
            zip_ref.extractall('/data/ScanNetV2/scans/scene0444_00')

if __name__=='__main__':
    unzip()
    #test()