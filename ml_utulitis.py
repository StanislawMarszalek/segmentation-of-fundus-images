import pickle
import datetime as dt
from copy import deepcopy
import numpy as np
import cv2
from image_processing import normalize01


#
#  Module to:
# - wrap functions from pickle to save and load machine_learning model
# - provide functions to extract feauters and lables from images
# -
#

def extract_features(frame:cv2.Mat|cv2.UMat)->list:
    """
    Extract features (mean, var, Hu moments) from the frame
    
    :param frame: Part of teh input images
    :type frame: cv2.Mat | cv2.UMat
    :return: List of features
    :rtype: list
    """
    features=[]
    frame_copy=deepcopy(frame)

    if frame.ndim > 2:

        frame_copy=cv2.cvtColor(frame_copy,cv2.COLOR_RGB2GRAY)


    frame_copy=normalize01(frame_copy)
    features.append(np.mean(frame_copy))
    features.append(np.var(frame_copy))

    moments = cv2.moments(frame_copy)
    hu = cv2.HuMoments(moments).flatten()
    features.extend(hu)
    hu_log = [-np.sign(h) * np.log10(abs(h) + 1e-12) for h in hu]
    features.extend(hu_log)

    #gradienty najwyzej usunąc
    #gx = cv2.Sobel(frame_copy, cv2.CV_64F, 1, 0, ksize=3)
    #gy = cv2.Sobel(frame_copy, cv2.CV_64F, 0, 1, ksize=3)

    #grad = np.sqrt(gx**2 + gy**2)

    #features.append(np.mean(grad))
    #features.append(np.var(grad))

    return features

def extract_label(frame,kernel_size:int)->int:
    #!!!numpy where must be applied to manual img before using ectract_lable function!!!
    if frame[kernel_size//2,kernel_size//2]==1:
        return 1

    else:
        return 0


def save_model(model,pathfile:str,tag:str|None=None)->None:

    if tag is None:
        date=dt.datetime.today()
        year=date.year
        month=date.month
        day=date.day
        hour=date.hour
        minutes=date.minute
        tag=f"{year}_{month}_{day}_{hour}_{minutes}"

    with open(f"{pathfile}_{tag}",mode="wb") as output_file:
        pickle.dump(model, output_file)

def load_model(pathfile:str):
    """
    Load model
    
    :param pathfile: Path to the model
    :type pathfile: str
    """
    with open(pathfile,mode="rb") as input_file:
        model=pickle.load(input_file)

    return model

if __name__=="__main__":
    example={"123":"test",
            123:"test"
            }

    save_model(example,"test","1_2_3_test")
    pathfile="test_1_2_3_test"
    model=load_model(pathfile)
    print(model==example)
