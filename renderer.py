import os
import sys
import torch
import numpy as np
import pytorch3d as t3d

from pytorch3d.io import load_ply

if __name__=='__main__':
    test_file = "scene0000_00_vh_clean.ply"
    vertices, faces = load_ply(test_file)
    print(vertices)
