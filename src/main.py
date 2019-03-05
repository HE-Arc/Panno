import cv2
import numpy

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

# FIXME
def analyse(path):
    img = cv2.imread(path)
    assert img is not None and img.size != 0, 'Invalid image'
    cv2.imshow('reference image',img)

    # Gray scale
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('reference image',imgGray)

    # Normalize
    imgNormalized = cv2.normalize(imgGray,None,0,255,cv2.NORM_MINMAX)
    cv2.imshow('imgNormalized',imgNormalized)

    # Edge detection
    imgEdges = cv2.Canny(imgNormalized,img.width,img.height)

    # Math morph
    kernel = np.ones((5,5),np.uint8)
    imgOpening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Edge 2.0
    imgGradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    # TODO: segmentation


# ___ MAIN ___
analyse('..\\images\\test.png')

cv2.waitKey(0)
cv2.destroyAllWindows()
