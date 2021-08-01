"""
Usage:
from utils.miou import miou_2d, miou_3d
"""
import torch
import numpy as np
import warnings

def miou_2d(pred, label):
    """
    Input two torch array, with shape [batch_size, width, length]
    Return MIOU, also a torch array
    """
    # ignore warning generated in computing IOU
    warnings.filterwarnings("ignore")
    if pred.device!='cpu':
        pred = pred.cpu()
        label = label.cpu()
    pred = pred.numpy().astype(np.int64)
    label = label.numpy().astype(np.int64)
    MIOU = []
    for single_image, single_label in zip(pred,label):# get one img
        
        max_class = max(np.unique(label).max(),np.unique(pred).max())
        confusion_matrix = np.zeros([max_class+1,max_class+1])
        for single_row_image, single_row_label in zip(single_image, single_label): # get each row
            # flatten
            single_row_image, single_row_label = single_row_image.flatten(), single_row_label.flatten()
            
            # bincount
            pred_count = np.bincount(single_row_image, minlength=max_class+1)
            label_count = np.bincount(single_row_label, minlength=max_class+1)
            
            # category
            category_count = single_row_label * (max_class+1) + single_row_image
            
            # construct confusion matrix
            confusion_matrix += np.bincount(category_count, minlength=(max_class+1)**2).reshape([max_class+1,max_class+1])
            
            
        
        Intersection = np.diag(confusion_matrix)
        Union = confusion_matrix.sum(axis=0) + confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        
        IOU = Intersection / Union
        MIOU.append(np.nanmean(IOU))
    
    MIOU = np.array(MIOU)
    return MIOU




def miou_3d(pred, label):
    """
    Input two torch array, with shape [point_num]
    Return MIOU, also a torch array
    """
    # ignore warning generated in computing IOU
    warnings.filterwarnings("ignore")

    if pred.device!='cpu':
        pred = pred.cpu()
        label = label.cpu()

    pred = pred.numpy().astype(np.int64)
    label = label.numpy().astype(np.int64)

    max_class = max(np.unique(label).max(),np.unique(pred).max())
    confusion_matrix = np.zeros([max_class+1,max_class+1])
      
    # bincount
    pred_count = np.bincount(pred, minlength=max_class+1)
    label_count = np.bincount(label, minlength=max_class+1)
            
    # category
    category_count = label * (max_class+1) + pred
            
    # construct confusion matrix
    confusion_matrix += np.bincount(category_count, minlength=(max_class+1)**2).reshape([max_class+1,max_class+1])
            
            
    # compute MIOU
    Intersection = np.diag(confusion_matrix)
    Union = confusion_matrix.sum(axis=0) + confusion_matrix.sum(axis=1) - np.diag(confusion_matrix) 
    IOU = Intersection / Union
    MIOU = np.nanmean(IOU)
    
    return MIOU





if __name__=='__main__':
    pred = torch.ones([2,128,128])
    label = torch.ones([2,128,128])
    label[0,100,100] = 5
    label[1,100,100] = 5
    miou_2d(pred, label)
