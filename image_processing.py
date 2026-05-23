import numpy as np
import matplotlib.pyplot as plt
from cv2 import medianBlur,imread,cvtColor,getStructuringElement,MORPH_ELLIPSE,morphologyEx,MORPH_CLOSE,Mat,UMat,COLOR_BGR2RGB


def show_img(image:Mat,title:str|None=None,gray_scale:bool=True)->None:
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

        img=cvtColor(img,COLOR_BGR2RGB)

    except (ValueError,FileNotFoundError,FileExistsError):
        print(f"Could not ope the file: {pathfile}")
        return None

    return img

def normalize01(image:Mat)->Mat:
    """
    Normalize given image
    
    :param image: Image to normalize
    :type image: Mat
    :return: Normalized version of the input image
    :rtype: Mat
    """
    return (image-image.min())/(image.max()-image.min()) if image.min()!=image.max() else 0



def preproces_pipeline(img:Mat,median_kernel_size:int=11)->Mat:

    """
    Preprocess pipeline
    
    :param img: Original image
    :type img: Mat
    :param median_kernel_size: Size of median blur kernel
    :type median_kernel_size: int
    :return: Modified image to extract data from
    :rtype: Mat
    """

    if img.ndim > 2:
        img = img[:, :, 1]   # green channel

    img = medianBlur(img, median_kernel_size)

    img = normalize01(img)
    return img

def postproces_pipeline(img:Mat,kernel_size:int=17)->Mat:
    """
    Postprocess pipeline to clean and fully reconstruct the image
    
    :param img: Image to modify
    :type img: Mat
    :param kernel_size: Size of filling kernel
    :type kernel_size: int
    :return: Description
    :rtype: Mat
    """
    #type must be unisgned short
    img = img.astype(np.uint8)
    #creating kernel
    kernel_fill = getStructuringElement(MORPH_ELLIPSE, (kernel_size, kernel_size))
    img=morphologyEx(img, MORPH_CLOSE, kernel_fill)
    #normalizing results
    img = normalize01(img)
    return img


def draw_vessels(original_img:Mat,found_vessels_img)->Mat:
    """
    Add found vessels to the original image
    
    :param original_img: Original image 
    :type original_img: Mat
    :param found_vessels_img: Vessels found in the original image
    :return: Original image with added vessels
    :rtype: Mat
    """

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
