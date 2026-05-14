from sklearn.ensemble import RandomForestClassifier
from image_processing import read_img,show_img
import numpy as np
import cv2
from copy import deepcopy
from skimage.feature import local_binary_pattern

def extract_features(frame)->list:
    features=[]
    frame_copy=deepcopy(frame)

    if frame.ndim > 2:
        
        frame_copy=cv2.cvtColor(frame_copy,cv2.COLOR_RGB2GRAY)

    ##Jezeli beda slabe wyniki dodac robienie preprocessingu przdd uzyciem exrtrac feature

     # CLAHE sprobowanie ,moze tutaj to cos da
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    frame_copy = clahe.apply(frame_copy)


    features.append(np.mean(frame_copy))
    features.append(np.var(frame_copy))
    
    moments = cv2.moments(frame_copy)
    hu = cv2.HuMoments(moments).flatten()
    features.extend(hu)
    
    ###sprobowac bez lbp zobaczcy jak pojdzie 
    #lbp = local_binary_pattern(frame_copy, P=8, R=1)

    #features.append(np.mean(lbp))
    #features.append(np.var(lbp))
    return features

def extract_label(frame,kernel_size:int)->int:
    if frame[kernel_size//2,kernel_size//2]==1:#pamietac zeby wherowac manuale
        return 1
    else:
        return 0