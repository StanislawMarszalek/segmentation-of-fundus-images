from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
from cv2 import Mat
import numpy as np
import matplotlib.pyplot as plt

def create_confusions_matrix(true_values:Mat ,predicted_values:Mat,
                            true_value:str="White",false_value:str="Black")->None:#tuple[tuple[int,int,int,int], ConfusionMatrixDisplay]:


    """
    Display confusions matrix and metrics
    
    :param true_values: Image that is considered to be true image of vessels
    :type img: Mat
    :param predicted_values: Image with predicted vessels
    :type mask: Mat
    :param true_value: Label for true values
    :type true_value: str
    :param flase_value: Label for false values
    :type threshold: str
    """

    matrix=confusion_matrix(true_values.flatten(),predicted_values.flatten())
    tn:int; fp:int; fn:int; tp:int
    displayed_matrix=ConfusionMatrixDisplay(matrix).from_predictions(true_values.flatten(),
                                                                    predicted_values.flatten(),display_labels=[false_value,true_value])


    tn, fp, fn, tp=matrix.ravel().tolist()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    #Unbalanced metrics
    balanced_acc = (sensitivity + specificity) / 2
    g_mean = np.sqrt(sensitivity * specificity)

    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Sensitivity (Czułość): {sensitivity:.4f}\n"
        f"Specificity (Swoistość): {specificity:.4f}\n"
        f"Balanced Acc: {balanced_acc:.4f}\n"
        f"G-Mean: {g_mean:.4f}\n"
    )

    plt.figtext(0.5, 0.01, metrics_text, ha="center", fontsize=10)
    plt.tight_layout(rect=[0, 0.2, 1, 1])
    plt.show()
    return #(tn,fp,fn,tp)#,displayed_matrix
    

if __name__=="__main__":
    data=np.array([[0,1,1],[1,0,0]])
    print(data.flatten())
    data2=np.array([[1,1,0],[0,0,0]])
    create_confusions_matrix(data,data2)
