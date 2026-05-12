import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv2 import GaussianBlur,createCLAHE,medianBlur,bitwise_and,imread,cvtColor

### PIPELINE:READ->EXTRACT GREEN->Gausian bluring/CLACHE-> [TRANSFORM TO 0-1 SCALE]
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
        #
        img=cvtColor(img,cv2.COLOR_BGR2RGB)
    except (ValueError,FileNotFoundError,FileExistsError):
        print(f"Could not ope the file: {pathfile}")
        return None

    return img

def normalize01(image:np.ndarray)->np.ndarray:
    return (image-image.min())/(image.max()-image.min())



def preproces_pipeline(img, clahe_clip=3.0, clahe_grid=(8, 8),clip=False
                       ,median_kernel_size=5, mask=None)->cv2.Mat:

    if img.ndim > 2:
        img = img[:, :, 1]   # green channel

    if mask is not None:
        if mask.ndim > 2:
            mask = mask[:, :, 1]
        img = img&mask

    if clip:
        img=np.clip(img,10,245)

    img = medianBlur(img, median_kernel_size)
    clahe = createCLAHE(clahe_clip, clahe_grid)
    img = clahe.apply(img)

    img = normalize01(img)
    return img



if __name__=="__main__":
    img=read_img(".\images\\01_dr.JPG")
    #EXTRAKCJA DO ZIELIONEGO
    #img=preproces_pipeline(img,2)
    show_img(img)
    #plt.subplot(121),plt.imshow(img),plt.title('Original')
    #plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
    #plt.xticks([]), plt.yticks([])
    #plt.show()
    
    #print(img.max())
