# Virtual Multi-view Fusion
My personal implementation of paper: Virtual Multi-view Fusion for 3D Semantic Segmentation (ECCV 2020)

# Usage
**First**, install packages this project depends on, including:
```
trimesh, pypng
```

**Second**, prepare **ScanNet** Dataset and change your own parameters in **`config/config.yaml`**.


**Third**, run the code.
Create virtual view imgs for training.
```
python create_img.py
```

Train & evaluate 2D image (Change mode in the code).
```
python pipeline_2d.py 
```

Do inference on 3D points with 2D fusion.
```
python pipeline_3d.py
```



# TODO
- [x] create img, label, depth from single point cloud

- [x] pose selection for single point cloud (handcraft selectoion however)

- [] a pipeline to train on generated 2d imgs

- [] inference


# 2D-3D Fusion Algorithm
1. Use 3D point, extrinsic and intrinsic, and get project point $P_{proj}$. 
2. Compute the theoretical depth prediction, based on 3D point, extrinsic and intrinsic. Denote as $D_{pred}$
3. Based on the size of the depth img, filter out the points not in the depth img. Also, filter out the depths. Get $P_{proj}^{bound}$ and $D_{pred}^{bound}$.
4. Depth Check. Get the real depth of each point in $P_{proj}^{bound}$ from the depth img, denoted as $D_{real}^{bound}$. Compare $D_{real}^{bound}$ and $D_{pred}^{bound}$ with the threshold $\delta$ and get mask $M_{satisf}$. 
5. Collect Features. Get features for points in $P_{proj}^{bound}[M_{satisfy}]$.

# Related Work
1. [Virtual Multi-view Fusion](https://arxiv.org/abs/2007.13138)
2. [ScanNet](https://github.com/ScanNet/ScanNet)
3. [DeepLabV3+](https://github.com/jfzhang95/pytorch-deeplab-xception)
4. [UNet](https://github.com/milesial/Pytorch-UNet)
