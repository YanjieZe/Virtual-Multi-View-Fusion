import numpy as np
import torch.utils.data as data
import os
import torch
from plyfile import PlyData

class VirtualviewScannetDataset(data.Dataset):
    """
    Dataset of Scenes, virtual view
    """
    def __init__(self, cfg, mode='train',use_transform=True):

        self.cfg = cfg
        self.use_transform = use_transform
        if mode=='train':
            self.root_path = cfg.dataset.train_path        
        elif mode=='test':
            self.root_path = cfg.dataset.test_path
        else:
            raise Exception("Mode Error: Only Train/Test Supported")

        self.mode = mode
        self.dir_list = os.listdir(self.root_path)
        
        self.mapping = self.get_vaild_class_mapping()


    def __len__(self):
        return len(self.dir_list)
    
    def __getitem__(self, index):
        scene_id = self.dir_list[index]

        # TODO: load virtual img
        # # get images, including color and depth
        # imgset = ImageDataset(cfg=self.cfg, 
        #                 root_path=self.root_path, 
        #                 scene_id=scene_id,
        #                 use_transform=self.use_transform) 
        
        # get mesh
        mesh_file_path = os.path.join(self.root_path, scene_id, "%s_vh_clean_2.ply"%(scene_id))
        agg_file_path = os.path.join(self.root_path, scene_id,"%s.aggregation.json"%(scene_id)) 
        seg_file_path = os.path.join(self.root_path, scene_id, "%s_vh_clean_2.0.010000.segs.json"%(scene_id))
        label_map_file_path = self.cfg.dataset.label_map

        axis_alignment = np.fromstring(open(os.path.join(self.root_path, scene_id, "%s.txt" % (scene_id))).readline().split('=')[1], sep=' \n\t', dtype=float).reshape(4, 4)
        
        mesh_vertices, semantic_label, instance_label = get_train_mesh(mesh_file_path,
                            agg_file_path,
                            seg_file_path,
                            label_map_file_path,
                            type='instance')
        mesh_vertices = torch.from_numpy(mesh_vertices.astype(np.float64))
        
        # process the semantic label into the benchmark label
        semantic_label = torch.from_numpy(semantic_label.astype(np.int32))
        
        valid_class_id = self.cfg.valid_class_ids
        mask = torch.zeros_like(semantic_label).bool()
        for class_id in valid_class_id:
            mask = mask | (semantic_label==class_id)
        
        semantic_label_ = torch.zeros_like(semantic_label)
        semantic_label_[mask] = semantic_label[mask]
        semantic_label = semantic_label_
        
        def idx_map(idx):
            return self.mapping[idx]
        semantic_label = torch.from_numpy(np.array(list(map(idx_map, semantic_label)))).long()
        
        # # get camera params
        # intrinsic_path = os.path.join(self.root_path, scene_id, "exported", "intrinsic")
        # intrinsic_color_matrix = self.get_intrinsic_matrix(os.path.join(intrinsic_path, "intrinsic_color.txt"))
        # intrinsic_depth_matrix = self.get_intrinsic_matrix(os.path.join(intrinsic_path, "intrinsic_depth.txt"))
        
        # # get align extrinsic
        # colorToDepthExtrinsics = self.get_align_extrinsic(scene_id)

        # alignment?
        # mesh_vertices = (mesh_vertices - axis_alignment[:3, -1]) @ axis_alignment[:3, :3]
        pc_xyz, pc_color = self.get_pointcloud(scene_id)
        pc_xyz, pc_color = torch.from_numpy(pc_xyz).double(), torch.from_numpy(pc_color).double()

        return { # 'imgset':imgset, 
                'point_cloud_xyz':pc_xyz,
                'point_cloud_color':pc_color,
                'mesh_vertices':mesh_vertices, 
                'semantic_label':semantic_label, 
                'instance_label':instance_label,
                # 'intrinsic_color':intrinsic_color_matrix,
                # 'intrinsic_depth':intrinsic_depth_matrix,
                # 'extrinsic_color_to_depth':colorToDepthExtrinsics
                }

    def get_pointcloud(self, scene_id):
        mesh_file_path = os.path.join(self.root_path, scene_id, "%s_vh_clean_2.ply"%(scene_id))
        # 1 get vertices
        with open(mesh_file_path, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
            vertices[:, 0] = plydata['vertex'].data['x']
            vertices[:, 1] = plydata['vertex'].data['y']
            vertices[:, 2] = plydata['vertex'].data['z']
            vertices[:, 3] = plydata['vertex'].data['red']
            vertices[:, 4] = plydata['vertex'].data['green']
            vertices[:, 5] = plydata['vertex'].data['blue']
        
        # 2   get axisAlignment
        meta_file = os.path.join(self.root_path, scene_id,'%s.txt'%(scene_id))
        lines = open(meta_file).readlines()
        axis_align_matrix = None
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]

        # 3. Offset point cloud PLY file
        if axis_align_matrix != None:
            axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4)) #  matrix
            pts = np.ones((vertices.shape[0], 4))
            pts[:, 0:3] = vertices[:, :3]
            pts = np.dot(pts, axis_align_matrix.transpose())
            aligned_vertices = np.copy(vertices)
            aligned_vertices[:, 0:3] = pts[:, 0:3]

        # 4. Re-write PLY
        points = aligned_vertices[:,:3]
        colors = aligned_vertices[:,3:6]
   
        return points, colors

    def get_vaild_class_mapping(self):
        valid_class_ids = self.cfg.valid_class_ids
        max_id = valid_class_ids[0]
        mapping = np.ones(max_id+1)*valid_class_ids.index(max_id)
        
        for i in range(max_id+1):
            if i in valid_class_ids:
                mapping[i] = valid_class_ids.index(i)
            else:
                mapping[i] = valid_class_ids.index(max_id)
        
        return mapping


    def get_intrinsic_matrix(self, file_path):
        """
        read the intrinsic matrix in file *.txt
        """
        with open(file_path, 'r') as f:
            matrix = [[float(num) for num in line.split(' ')] for line in f]
        matrix = torch.from_numpy(np.array(matrix))
        return matrix

    def get_align_extrinsic(self,scene_id):
        """
        Read colorToDepthExtrinsics in .txt file

        TODO: Check whether we extract the extrinsic correctly.
        """
        file_path = os.path.join(self.root_path, scene_id,'%s.txt'%(scene_id))
        target_line = None
        with open(file_path, 'r') as open_file:
            for line in open_file:
                if line.split()[0]=='colorToDepthExtrinsics':
                    target_line = line
                    break
        target_line=target_line.split()
        del target_line[0]
        del target_line[0]
        extrinsic_matrix = torch.zeros(4,4).float()
        
        for i in range(4):
            for j in range(4):
                extrinsic_matrix[i][j] = float(target_line[i*4+j])
        return extrinsic_matrix


