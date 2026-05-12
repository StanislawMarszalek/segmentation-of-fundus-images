import numpy as np
import matplotlib.pyplot as plt

from cv2 import GaussianBlur,createCLAHE,medianBlur

### PIPELINE:READ->EXTRACT GREEN->Gausian bluring/CLACHE-> [TRANSFORM TO 0-1 SCALE]
#ZROBIC FUNCKJE PIPELINE KOTRA BEDZIE PO KOLEJI WYKONYWAC
def show_img(image:np.ndarray,title:str|None=None)->None:
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


def read_img(pathfile:str)->np.ndarray:
    """
    Read and return an image
    
    :param pathfile: Description
    :type pathfile: str
    :return: Description
    :rtype: UMat | None
    """
    try:
        img=plt.imread(pathfile)
        assert img is not None ,"Image cant be None"
    except (ValueError,FileNotFoundError,FileExistsError):
        print(f"Could not ope the file: {pathfile}")
        return None

    return img

def normalize01(image:np.ndarray)->np.ndarray:
    return (image-image.min())/(image.max()-image.min())



def preproces_pipeline(img:np.ndarray,clahe_clip:float=3.0,
                        clahe_grid:tuple[int, int]=(8,8), gaussian_grid:tuple[int,int]=(5,5),
                        gauss_x_std:float=0)->np.ndarray:


    #Extracting green channel
    if len(img.shape)>1:
        img=img[:,:,1]
    
    img=medianBlur(img,5)
    img=GaussianBlur(img,gaussian_grid,gauss_x_std)
    #Clahing
    clahe=createCLAHE(clahe_clip,clahe_grid)
    img=clahe.apply(img)
    
    
    #Normalizing
    img=normalize01(img)
    return img



if __name__=="__main__":
    img=read_img(".\images\\01_dr.JPG")
    #EXTRAKCJA DO ZIELIONEGO
    img=preproces_pipeline(img,2)
    show_img(img)
    #plt.subplot(121),plt.imshow(img),plt.title('Original')
    #plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
    #plt.xticks([]), plt.yticks([])
    #plt.show()
    
    #print(img.max())
