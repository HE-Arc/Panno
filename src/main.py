import cv2
import numpy as np
from matplotlib import pyplot as plt

###############################
# Script to detect road signs #
# --------------------------- #
# Steps :                     #
#     - Grayscale             #
#     - Edge detection        #
#     - Math. morph.          #
#     - Segmentation          #
#                             #
# Inspired from :
#     - : https://www.researchgate.net/publication/259738019_Optimized_Method_for_Iranian_Road_Signs_Detection_and_Recognition_System
#     - : https://www.researchgate.net/publication/281368553_Traffic_sign_recognition_without_color_information
###############################

def show(img):
    pltImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(pltImage)
    plt.show()

# FIXME
def analyse(path):
    img = cv2.imread(path, 0)
    img = cv2.medianBlur(img, 5)
    assert img is not None and img.size != 0, 'Invalid image'
    
    # Normalize
    imgNormalized = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)

    # Math morph
    kernel = np.ones((3,3),np.uint8)

    # Dilate normalized image, try also with erode
    dilate = cv2.dilate(imgNormalized,kernel,iterations = 1)

    # Ouverture
    imgOpening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel)
    
    show(imgOpening)

    # Canny (Edge)
    imgEdges = cv2.Canny(imgOpening, 150 , 220)

    show(imgEdges)

    # Edge Fermeture
    imgGradient = cv2.morphologyEx(imgEdges, cv2.MORPH_GRADIENT, kernel)

    # Affichage
    show(imgGradient)

    # TODO: segmentation


# ___ MAIN ___
analyse('C:\\Dev\\!Traitement_Images\\Panno\\images\\test.png')
analyse('C:\\Dev\\!Traitement_Images\\Panno\\images\\many.jpg')
analyse('C:\\Dev\\!Traitement_Images\\Panno\\images\\autobahn.png')
analyse('C:\\Dev\\!Traitement_Images\\Panno\\images\\yellow_roadsign.jpg')

cv2.waitKey(0)
cv2.destroyAllWindows()
