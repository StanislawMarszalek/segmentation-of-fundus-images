import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv2 import medianBlur,imread,cvtColor,getStructuringElement,MORPH_ELLIPSE,morphologyEx,MORPH_CLOSE,Mat,UMat


def show_img(image:cv2.Mat,title:str|None=None,gray_scale:bool=True)->None:
    """
    Display a given image in the gray scale
    
    :param image: Image to display
    :type image: np.ndarray
    :param title: Title for the image , if None no title is displayed
    :type title: str | None
    """
    if isinstance(title,str):
        plt.title(title)
    if gray_scale:
        plt.imshow(image,cmap="gray")
    else:
        plt.imshow(image)
    plt.show()
    return


def read_img(pathfile:str)->UMat|Mat:
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

def normalize01(image:Mat)->Mat:
    return (image-image.min())/(image.max()-image.min()) if image.min()!=image.max() else 0



def preproces_pipeline(img:Mat,median_kernel_size:int=11)->Mat:

    if img.ndim > 2:
        img = img[:, :, 1]   # green channel

    img = medianBlur(img, median_kernel_size)

    img = normalize01(img)
    return img

def postproces_pipeline(img:Mat,kernel_size:int=17)->Mat:
    #type must be unisgned short
    img = img.astype(np.uint8)
    #creating kernel
    kernel_fill = getStructuringElement(MORPH_ELLIPSE, (kernel_size, kernel_size))
    img=morphologyEx(img, MORPH_CLOSE, kernel_fill)
    #normalizing results
    img = normalize01(img)
    return img


def draw_vessels(original_img:Mat,found_vessels_img)->Mat:
    for x in range(original_img.shape[0]):
        for y in range(original_img.shape[1]):
            if found_vessels_img[x,y]==1:
                original_img[x,y,:]=(0,255,0)
    return original_img


if __name__=="__main__":
    img=read_img(".\images\\01_dr.JPG")
    show_img(img,"Before processing")
    img=preproces_pipeline(img,11)
    show_img(img,"After pre-procesing")

