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

# TODO: A tester : Prendre l'image en 4k, puis la réduire en taille par 2, 4, etc
# TODO: Appliquer les algos sur des zones de couleurs vives
# FIXME: Améliorer avec des analyses par couleur, exemple : Hough sur le bleu

def hough_lines(img_src, img_dest):
    lines = cv2.HoughLines(img_src,1,np.pi/180.0, 150)
    ''' 
    4th parameter : Threshold
    5th parameter : Minimum length of line
    6th parameter : Maximum allowed gap between line segments to treat them as single line
    '''
 
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img_dest,(x1,y1),(x2,y2),(255,0,255),5)

    return img_dest

def hough_lines_p(img_src, img_dest, minimum_length, max_gap):
    lines = cv2.HoughLinesP(img_src,1,np.pi/180.0, 150, minimum_length, max_gap)
    ''' 
    4th parameter : Threshold
    5th parameter : Minimum length of line
    6th parameter : Maximum allowed gap between line segments to treat them as single line
    '''
    for var in lines:
        x1 = var[0][0]
        y1 = var[0][1]
        x2 = var[0][2]
        y2 = var[0][3]

        cv2.line(img_dest,(x1,y1),(x2,y2),(255,0,255),5)
    
    return img_dest

def show(img):
    pltImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(pltImage)
    plt.show()

# FIXME
def analyse(path):
    img = cv2.imread(path)
    assert img is not None and img.size != 0, 'Invalid image'

    height, width, channels = img.shape
    print(width)
    print(height)

    # Convert it to gray
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize
    imgNormalized = cv2.normalize(imgGray, None, 0, 255,cv2.NORM_MINMAX)

    # Blur
    imgBlured = cv2.medianBlur(imgNormalized, 5)

    # Math morph
    kernel = np.ones((3,3),np.uint8)
    #TODO: vérifier la taille du noyau pour les différentes tailles

    # Dilate normalized image, try also with erode
    dilate = cv2.dilate(imgBlured,kernel,iterations = 1)

    # Ouverture
    imgOpening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel)
    
    # Canny (Edge)
    imgEdges = cv2.Canny(imgOpening, 150, 220)

    # Edge Fermeture
    imgGradient = cv2.morphologyEx(imgEdges, cv2.MORPH_GRADIENT, kernel)

    # Show edges
    show(imgGradient)

    # Probabilistic Hough
    minimum_length = width/100.0
    max_gap = minimum_length - minimum_length * 0.25

    img = hough_lines_p(imgGradient, img, minimum_length, max_gap)

    # Hough Circles
    # FIXME: Trouver des bon ratio
    circles = cv2.HoughCircles(imgGradient, cv2.HOUGH_GRADIENT, 1.1, 35)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
    
        for (x, y, r) in circles:
            # draw the outer circle
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)


    # Affichage
    show(img)

    # TODO: segmentation


# ___ MAIN ___
analyse('C:\\Dev\\!Traitement_Images\\Panno\\images\\test.png')
analyse('C:\\Dev\\!Traitement_Images\\Panno\\images\\many.jpg')
analyse('C:\\Dev\\!Traitement_Images\\Panno\\images\\autobahn.png')
analyse('C:\\Dev\\!Traitement_Images\\Panno\\images\\yellow_roadsign.jpg')

cv2.waitKey(0)
cv2.destroyAllWindows()
