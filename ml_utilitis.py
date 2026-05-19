import pickle
import datetime as dt
from copy import deepcopy
import numpy as np
import cv2
from image_processing import normalize01


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Module to:                                                           #
# - wrap functions from pickle to save and load machine_learning model  #
# - provide functions to extract feauters and lables from images        #
# - povide functions to clean and process images after classification   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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


    return features

def extract_label(frame,kernel_size:int)->int:
    """
    Extract label from the frame center
    
    :param frame: Parto of image to extract label from
    :param kernel_size: Description
    :type kernel_size: Size of sliding window
    :return: Description
    :rtype: int
    """
    #!!!numpy where must be applied to manual img before using ectract_lable function!!!
    if frame[kernel_size//2,kernel_size//2]==1:
        return 1

    else:
        return 0

def remove_small_components(img_to_clean, min_size:int=60):

    """
    Remove small componnets from the image
    
    :param img_to_clean: Image to remove the noise from
    :param min_size: Description
    :type min_size: int
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        img_to_clean.astype(np.uint8),
        connectivity=8
    )

    cleaned = np.zeros_like(img_to_clean)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_size:
            cleaned[labels == i] = 1

    return cleaned


def predict_full_image( img, mask, classifier,
    preprocess_function, kernel_size: int = 5, use_gpu:bool=False )->np.ndarray:

    """
    Apply preprocess function then make prediction
    
    :param img: Image to extract vessels from
    :param mask: Field of view (FOV) mask
    :param classifier: Model to do prediction (must provide `predict` function)
    :param preprocess_function: Function to preprocess the image
    :param kernel_size: Size of slicing kernel
    :type kernel_size: int
    :param use_gpu: Switch to decide if use GPU ( `classifier` must provide an accurate interface )
    :type use_gpu: bool
    :return: Processed image with found vessels
    :rtype: ndarray
    """

    half = kernel_size // 2

    # image preprocessing
    proc_img = preprocess_function(img)

    if proc_img.ndim > 2:
        proc_img = cv2.cvtColor(proc_img, cv2.COLOR_RGB2GRAY)

    # converting mask (FOV) to 0 and 1 
    if mask.ndim > 2:
        mask = mask[:, :, 1]
    mask = np.where(mask > 0, 1, 0).astype(np.uint8)

    # image padding
    padded_img = np.pad(proc_img, pad_width=half, mode='reflect')

    height, width = proc_img.shape
    pred_mask = np.zeros((height, width), dtype=np.uint8)

    #variables for data
    features_list = []
    coords = []

    for x in range(height):
        for y in range(width):

            # Only points inside the FOV mask are important
            if mask[x, y] == 0:
                continue

            patch = padded_img[x:x + kernel_size, y:y + kernel_size]

            if patch.shape[0] != kernel_size or patch.shape[1] != kernel_size:
                continue

            features = extract_features(patch)
            features_list.append(features)
            coords.append((x, y))

    # one predictio  for whole data
    features_array = np.asarray(features_list, dtype=np.float32)
    if use_gpu:
        predictions = classifier.predict(features_array,task_type="GPU")
    else:
        predictions = classifier.predict(features_array)

    # savinig prediction
    for (x, y), pred in zip(coords, predictions):
        pred_mask[x, y] = pred
    pred_mask&=mask

    return pred_mask

def save_model(model,pathfile:str,tag:str|None=None)->None:
    """
    Save model
    
    :param model: Data (model) to be saved
    :param pathfile: Path to place where model will be stored (with added tag)
    :type pathfile: str
    :param tag: Tag to specify version of model (if not given data is used)
    :type tag: str | None
    """
    if tag is None:
        date=dt.datetime.today()
        year=date.year
        month=date.month
        day=date.day
        hour=date.hour
        minutes=date.minute
        tag=f"{year}_{month}_{day}_{hour}_{minutes}"

    with open(f"{pathfile}_{tag}.pkl",mode="wb") as output_file:
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
    pathfile="test_1_2_3_test.plk"
    model=load_model(pathfile)
    print(model==example)
