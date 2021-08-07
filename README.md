# Virtual Multi-view Fusion
My personal implementation of paper: Virtual Multi-view Fusion for 3D Semantic Segmentation (ECCV 2020)

# Usage
**First**, install packages this project depends on, including:
```
trimesh, pypng
```

**Second**, prepare **ScanNet** Dataset and change your own parameters in **`config/config.yaml`**.


**Third**, run the code.

Train & evaluate 2D image (Change mode in the code).
```
python pipeline_2d.py 
```

Do inference on 3D points with 2D fusion.
```
python pipeline_3d.py
```

Create virtual view imgs.
```
python create_img.py
```

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

# Problems
1. 2D 模型在第一个epoch的loss下降，后来就不下降了
2. 2D 模型只能处理较小图片（128\*128），无法处理原尺寸的图片（480\*640）。（训练和测试的时候都只能resize）
3. fusion算法投影坐标还有点问题，depth check的差距比较大，需要threshold设置的很大才有用。
4. 结合fusion算法在3D做inference的时候，不能直接用原图片，会cuda out of memory。只能先把图片resize再做。如何处理？因此导致了目前inference其实是不对的。
5. virtual view差不多完成了，现在关键是怎么找到角度。