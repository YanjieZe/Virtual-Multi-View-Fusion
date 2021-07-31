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

    pred = pred.numpy().astype(np.int64)
    label = label.numpy().astype(np.int64)
    MIOU = []
    for single_image, single_label in zip(pred,label):# get one img
        
        max_class = np.unique(single_label).max()
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


def _fast_hist(row_label, row_image, n_class):
   
    mask = (row_label>= 0) & (row_label < n_class)
    hist = np.bincount(
        n_class * row_label[mask].astype(int) +
        row_image[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def miou_3d(pred, label):
    pass

if __name__=='__main__':
    pred = torch.ones([2,128,128])
    label = torch.ones([2,128,128])
    label[0,100,100] = 5
    label[1,100,100] = 5
    miou_2d(pred, label)
