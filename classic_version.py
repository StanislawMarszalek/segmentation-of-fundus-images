from image_processing import preproces_pipeline,postproces_pipeline
from skimage.filters import sato
from cv2 import Mat
from numpy import where


def classic_vessel_segmentation(img:Mat,mask:Mat|None=None,median_kernel_size:int=11,
                    threshold:float=0.0175,filling_kernel_size:int=17):

    """
    Classic none machine/deep learning vessels segmentation algorithm
    
    :param img: Input image to do vessels segmentation
    :type img: Mat
    :param mask: Field of View mask, if not give it's not applied
    :type mask: Mat | None
    :param median_kernel_size: Size of median blur kernel
    :type median_kernel_size: int
    :param threshold: Threshold to decide if a point is white or black
    :type threshold: float
    :param filling_kernel_size: Size of filling kernel to close vessels
    :type filling_kernel_size: int
    """
    
    vessels=preproces_pipeline(img,median_kernel_size)
    vessels=sato(vessels,[1,2,3,4,5])
    vessels=where(vessels>threshold,1,0)
    if mask is not None:
        if len(mask.shape)>1:
            mask=mask[:,:,1]
        vessels&=mask
    vessels=postproces_pipeline(vessels,filling_kernel_size)

    return vessels
