from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
from cv2 import Mat
import numpy as np
import matplotlib.pyplot as plt

def create_confusions_matrix(true_values:Mat ,predicted_values:Mat,
                            true_value:str="White",flase_value:str="Black")->tuple[tuple[int,int,int,int], ConfusionMatrixDisplay]:
    
    matrix=confusion_matrix(true_values.flatten(),predicted_values.flatten())
    tn:int; fp:int; fn:int; tp:int
    displayed_matrix=ConfusionMatrixDisplay(matrix).from_predictions(true_values.flatten(),predicted_values.flatten(),
                                                                    display_labels=[flase_value,true_value])

    tn, fp, fn, tp=matrix.ravel().tolist()
    return (tn,fp,fn,tp),displayed_matrix
    

if __name__=="__main__":
    data=np.array([[0,1,1],[1,0,0]])
    print(data.flatten())
    data2=np.array([[1,1,0],[0,0,0]])
    create_confusions_matrix(data,data2)