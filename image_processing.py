import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv2 import medianBlur,imread,cvtColor,getStructuringElement,MORPH_ELLIPSE,morphologyEx,MORPH_CLOSE
from skimage.filters import frangi,sato,unsharp_mask

### PIPELINE:READ->EXTRACT GREEN->Gausian bluring-> [TRANSFORM TO 0-1 SCALE]
#ZROBIC FUNCKJE PIPELINE KOTRA BEDZIE PO KOLEJI WYKONYWAC
def show_img(image:cv2.Mat,title:str|None=None)->None:
    """
    Display a given image in the gray scale
    
    :param image: Image to display
    :type image: np.ndarray
    :param title: Title for the image , if None no title is displayed
    :type title: str | None
    """
    if isinstance(title,str):
        plt.title(title)
    plt.imshow(image,cmap="gray")
    plt.show()
    return


def read_img(pathfile:str)->cv2.UMat|cv2.Mat:
    """
    Read and return an image
    
    :param pathfile: Description
    :type pathfile: str
    :return: Description
    :rtype: UMat | None
    """
    try:
        img=imread(pathfile)
        if img is None:
            raise ValueError("Img not found")
        
        img=cvtColor(img,cv2.COLOR_BGR2RGB)
    except (ValueError,FileNotFoundError,FileExistsError):
        print(f"Could not ope the file: {pathfile}")
        return None

    return img

def normalize01(image:np.ndarray)->np.ndarray:
    return (image-image.min())/(image.max()-image.min())



def preproces_pipeline(img,median_kernel_size=11)->cv2.Mat:

    if img.ndim > 2:
        img = img[:, :, 1]   # green channel

    img = medianBlur(img, median_kernel_size)

    img = normalize01(img)
    return img

def postproces_pipeline(img)->cv2.Mat:
    img = img.astype(np.uint8)
    kernel_fill = getStructuringElement(MORPH_ELLIPSE, (15, 15))
    return morphologyEx(img, cv2.MORPH_CLOSE, kernel_fill)

if __name__=="__main__":
    img=read_img(".\images\\01_dr.JPG")
    show_img(img)

