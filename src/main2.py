import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

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

# RGB, HSV,
# Multi-résolution, si on trouve un élément à la même place pour chaque résolution on est sur qu'il existe

currently_processed_img_name = ''

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def resize(imglist, scale):
    l = []
    for img in imglist:
        (h, w) = img.shape[:2]
        l.append(cv2.resize(img, (int(w*scale), int(h*scale))))
    return l

def cut_green(img, criteria="max"):
    new_img = img.copy()
    (h, w) = new_img.shape[:2]
    for y in range(0, h):
       for x in range(0, w):
           r = int(new_img[y][x][2])
           g = int(new_img[y][x][1])
           b = int(new_img[y][x][0])
           if (2 * g) > (r + b):
               grey = max(r,g,b)
               if criteria == "min":
                   grey = min(r,g,b)
               elif criteria == "avg":
                    grey = np.uint8(np.mean([r,g,b]))
               new_img[y][x][0] = grey
               new_img[y][x][1] = grey
               new_img[y][x][2] = grey
    return new_img

def bilateral_filter(img):
    (h, w) = img.shape[:2]
    return cv2.bilateralFilter(img,30,50,50)

def save(img, path, name):
    global currently_processed_img_name
    path = currently_processed_img_name + path

    if not os.path.exists(path):
        os.makedirs(path)
    #pltImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # used for ploting
    cv2.imwrite(path + name +'.png', img)

def hough_lines(img_src, original):
    lines = cv2.HoughLines(img_src,1,np.pi/180.0, 50)
    '''
    4th parameter : Threshold
    5th parameter : Minimum length of line
    6th parameter : Maximum allowed gap between line segments to treat them as single line
    '''

    img_dest = original[:,:].copy()

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

def hough_lines_p(img_src, original, minimum_length, max_gap):
    lines = cv2.HoughLinesP(img_src,1,np.pi/180.0, 25, minimum_length, max_gap)
    '''
    4th parameter : Threshold
    5th parameter : Minimum length of line
    6th parameter : Maximum allowed gap between line segments to treat them as single line
    '''

    img_dest = original[:,:].copy()
    for var in lines:
        x1 = var[0][0]
        y1 = var[0][1]
        x2 = var[0][2]
        y2 = var[0][3]

        cv2.line(img_dest,(x1,y1),(x2,y2),(255,0,255),5)

    return img_dest

def hough_circles(img_src, original, index):
    global currently_processed_img_name

    img_dest = original[:,:].copy()
    (h, w) = img_dest.shape[:2]
    max_radius = int(min(w,h)/3)
    # FIXME: Trouver des bon ratio
    circles = cv2.HoughCircles(img_src, cv2.HOUGH_GRADIENT, 3.0, max_radius, maxRadius=max_radius)

    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # Fetch circles
    idx = 0
    for (x, y, r) in circles:

        # Crop circle
        crop_img = original[y-r:y+r, x-r:x+r].copy()

        # Export circle
        save(crop_img, f"{index}/", f"circle-{idx}")

        # draw the outer circle
        cv2.circle(img_dest, (x, y), r, (0, 255, 0), 4)

        # Increment index
        idx = idx + 1

    return img_dest

def contour(img_src, original, index, min_size):
    contours, hierarchy = cv2.findContours(img_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_dest = original[:,:].copy()

    idx = 0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        if w < min_size or h < min_size:
            continue

        cv2.rectangle(img_dest,(x,y),(x+w,y+h),(0,255,0),1)

        # Crop circle
        crop_img = original[y:y+h, x:x+w].copy()

        # Export contours
        save(crop_img, f"{index}/", f"contour-{idx}")

        # Increment index
        idx = idx + 1

    return img_dest

# FIXME
def analyse(imglist, name):
    i = 0
    for img in imglist:
        global currently_processed_img_name

        (h, w) = img.shape[:2]
        print(f"image {i}:{w}x{h}")

        # set name
        currently_processed_img_name = name
        save(img, f"{i}/", "0-original")


        img2 = bilateral_filter(cut_green(img, criteria="max"))
        save(img2, f"{i}/", "1-cut_and_bilblur")

        b,g,r = cv2.split(img)
        r = cv2.subtract(cv2.subtract(r,b),g)
        b = cv2.subtract(cv2.subtract(b,r),g)
        save(r, f"{i}/", "2-RED")
        save(b, f"{i}/", "2-BLUE")

        retval, r_T = cv2.threshold(r, 50, 255, cv2.THRESH_BINARY)
        save(r_T, f"{i}/", "3-threshold-RED")
        retval, b_T = cv2.threshold(b, 50, 255, cv2.THRESH_BINARY)
        save(b_T, f"{i}/", "3-threshold-BLUE")

        img_redblue = cv2.add(r_T,b_T)
        save(img_redblue, f"{i}/", "4-redblue")

        # Math morph
        kernel = np.ones((3,3),np.uint8)

        # Dilate normalized image, try also with erode
        img2 = cv2.dilate(img_redblue,kernel,iterations = 3)
        save(img2, f"{i}/", "5-dilate")

        # Ouverture
        img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
        save(img2, f"{i}/", "6-morphologyEx")

        # Canny (Edge)
        img2 = cv2.Canny(img2, 150, 220)
        save(img2, f"{i}/", "7-Canny")

        # Edge Fermeture
        img2 = cv2.morphologyEx(img2, cv2.MORPH_GRADIENT, kernel)
        save(img2, f"{i}/", "8-morphologyEx")

        # Probabilistic Hough
        minimum_length = min(w,h)/25.0
        max_gap = minimum_length - minimum_length * 0.25
        try:
            img_hough_lines = hough_lines_p(img2, img, minimum_length, max_gap)
            save(img_hough_lines, f"{i}/", "9-hough_lines")
        except:
            print(f"Can't process HoughLines for image {i}")

        # Contour
        try:
            img_contour = contour(img2, img, i, minimum_length)
            save(img_contour, f"{i}/", "10-contour")
        except:
            print(f"Can't process contour for image {i}")

        # Hough Circles
        try:
            img_hough_circles = hough_circles(img2, img, i)
            save(img_hough_circles, f"{i}/", "11-hough_circles")
        except:
            print(f"Can't process HoughCircles for image {i}")

        i += 1


# ___ MAIN ___

original_images = load_images_from_folder("images/true")
assert original_images is not None and len(original_images) != 0, 'Invalid image'
analyse(original_images, './output/test_full/')

cv2.destroyAllWindows()
