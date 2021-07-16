# Virtual Multi-view Fusion
My personal implementation of paper: Virtual Multi-view Fusion for 3D Semantic Segmentation (ECCV 2020)

# Information Log
root data path: /data/ScanNetV2

train data: /data/ScanNetV2/scans

from scene0000_00 to scene0706_00

test data: /data/ScanNetV2/scans_test


# Process Log
2021.7.6 

Begin to render. install pytorch 3d, cost a long time.

Success in a new enviroment **pytorch3d**.

# ScanNet
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
