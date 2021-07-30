import torch
import numpy as np

def miou_2d(pred, label):
    """
    Input two torch array, with shape [batch_size, width, length]
    Return MIOU, also a torch array
    """
    
    pred = pred.numpy().astype(np.int64)
    label = label.numpy().astype(np.int64)
    hist = None
    iou = None
    miou = torch.zeros(label.shape[0])
    i=0
    for single_image,single_label in zip(pred,label):
        count = 0
        # get one label's class num
        num_class = len(np.unique(single_label))
        
        for row_image,row_label in zip(single_image,single_label):
            if hist is None:
                hist = _fast_hist(row_label.flatten(), row_image.flatten(), num_class)
            else: hist += _fast_hist(row_label.flatten(), row_image.flatten(), num_class)
            
            count += 1
            
            if iou is None:
                iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)+0.000001)
            else: 
                iou += np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)+0.000001)
        
        # get one img's miou
        iou = iou/count
        
        miou[i] = iou.sum()/num_class
        i += 1
    
    return miou


def _fast_hist(row_label, row_image, n_class):
   
    mask = (row_label>= 0) & (row_label < n_class)
    hist = np.bincount(
        n_class * row_label[mask].astype(int) +
        row_image[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def miou_3d(pred, label):
    pass

if __name__=='__main__':
    pred = torch.ones([2,512,512])
    label = torch.ones([2,512,512])
    label[0,200,200] = 5
    label[0,100,100] = 10
    miou_2d(pred, label)
