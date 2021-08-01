# Virtual Multi-view Fusion
My personal implementation of paper: Virtual Multi-view Fusion for 3D Semantic Segmentation (ECCV 2020)

# Usage

**Change your own parameters in `config/config.yaml`**.

train & evaluation 2D image
```
python pipeline.py 
```


# 2D-3D Fusion Algorithm
1. Use 3D point, extrinsic and intrinsic, and get project point $P_{proj}$. 
2. Compute the theoretical depth prediction, based on 3D point, extrinsic and intrinsic. Denote as $D_{pred}$
3. Based on the size of the depth img, filter out the points not in the depth img. Also, filter out the depths. Get $P_{proj}^{bound}$ and $D_{pred}^{bound}$.
4. Depth Check. Get the real depth of each point in $P_{proj}^{bound}$ from the depth img, denoted as $D_{real}^{bound}$. Compare $D_{real}^{bound}$ and $D_{pred}^{bound}$ with the threshold $\delta$ and get mask $M_{satisf}$. 
5. Collect Features. Get features for points in $P_{proj}^{bound}[M_{satisfy}]$.


# ScanNet Dataset
```
<scanId>
|-- <scanId>.sens
    RGB-D sensor stream containing color frames, depth frames, camera poses and other data
|-- <scanId>_vh_clean.ply
    High quality reconstructed mesh
|-- <scanId>_vh_clean_2.ply
    Cleaned and decimated mesh for semantic annotations
|-- <scanId>_vh_clean_2.0.010000.segs.json
    Over-segmentation of annotation mesh
|-- <scanId>.aggregation.json, <scanId>_vh_clean.aggregation.json
    Aggregated instance-level semantic annotations on lo-res, hi-res meshes, respectively
|-- <scanId>_vh_clean_2.0.010000.segs.json, <scanId>_vh_clean.segs.json
    Over-segmentation of lo-res, hi-res meshes, respectively (referenced by aggregated semantic annotations)
|-- <scanId>_vh_clean_2.labels.ply
    Visualization of aggregated semantic segmentation; colored by nyu40 labels (see img/legend; ply property 'label' denotes the ScanNet label id)
|-- <scanId>_2d-label.zip
    Raw 2d projections of aggregated annotation labels as 16-bit pngs with ScanNet label ids
|-- <scanId>_2d-instance.zip
    Raw 2d projections of aggregated annotation instances as 8-bit pngs
|-- <scanId>_2d-label-filt.zip
    Filtered 2d projections of aggregated annotation labels as 16-bit pngs with ScanNet label ids
|-- <scanId>_2d-instance-filt.zip
    Filtered 2d projections of aggregated annotation instances as 8-bit pngs
```

# Related Work
1. [Virtual Multi-view Fusion](https://arxiv.org/abs/2007.13138)
2. [ScanNet](https://github.com/ScanNet/ScanNet)
3. [DeepLabV3+](https://github.com/jfzhang95/pytorch-deeplab-xception)
4. [UNet](https://github.com/milesial/Pytorch-UNet)

# Problems
1. 2D 模型在第一个epoch的loss下降，后来就不下降了
2. 2D 模型只能处理较小图片（128\*128），无法处理原尺寸的图片（480\*640）
3. fusion算法投影坐标还有点问题，depth check的差距比较大，需要threshold设置的很大才有用。
4. 结合fusion算法在3D做inference的时候，不能直接用原图片，会cuda out of memory。只能加transform再做？