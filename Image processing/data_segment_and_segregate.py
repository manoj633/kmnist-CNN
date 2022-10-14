"""""" """
--- This program reads the errors.txt file and converts the error points into a list.
--- Then reads the processed images and takes each character one by one
--- and sees if it is erroraneous
--- If yes, it discards it by saving in a different folder named set_of_Errors.
--- If no, it crops the 100x100 image to only contain the character and resize it to 28x28
--- Then inverts the image and saves each 28x28 pixel image as jpeg in the drive
--- (This program runs in Google Colaboratory)
"""

import cv2
import numpy as np


def crop(img):
    contours, h = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # cnt_img = contours[0]
    cc = 1
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 100 and w > 30 and h < 100 and h > 30:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cc += 1
    # cv2.imshow('img',cv2.resize(img,(0,0),fx=0.3,fy=0.3))
    return img, cc


def border(img):
    y = 0
    yph = img.shape[0]

    found_up = False
    found_down = False
    for i in range(0, img.shape[0]):  # rows y ver
        up_large = 0
        down_large = 0
        for j in range(0, img.shape[1]):  # columns x hor
            if img[i][j] > up_large:
                up_large = img[i][j]
            if img[img.shape[0] - i - 1][j] > down_large:
                down_large = img[img.shape[0] - i - 1][j]

        if found_up == False and up_large > 100:
            y = i
            found_up = True

        if found_down == False and down_large > 100:
            yph = img.shape[0] - i - 1
            found_down = True

    x = 0
    xpw = 0
    found_up = False
    found_down = False
    for j in range(0, img.shape[1]):  # rows y ver
        up_large = 0
        down_large = 0
        for i in range(0, img.shape[0]):  # columns x hor
            if img[i][j] > up_large:
                up_large = img[i][j]
            if img[i][img.shape[1] - j - 1] > down_large:
                down_large = img[i][img.shape[1] - j - 1]

        if found_up == False and up_large > 100:
            x = j
            found_up = True

        if found_down == False and down_large > 100:
            xpw = img.shape[1] - j - 1
            found_down = True

    final = np.zeros((28, 28, 1), np.uint8)
    resized = cv2.resize(img[y:yph, x:xpw], (26, 26))
    for i in range(resized.shape[0]):
        for j in range(resized.shape[1]):
            final[i + 1][j + 1] = resized[i][j]
    return final


def cut_img_contour(img, x, y, hh_img, c):
    try:
        contours, h = cv2.findContours(
            hh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours, h = cv2.findContours(
            hh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours, h = cv2.findContours(
            hh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        xc, yc, wc, hc = cv2.boundingRect(contours[0])
        # cv2.rectangle(img,(i*100+xc,j*100+yc),(i*100+xc+wc,j*100+yc+hc),(255,255,255),2)
        pix_img = border(hh_img[yc : yc + hc, xc : xc + wc])  # contour img
        pix_img = cv2.bitwise_not(pix_img, pix_img)
        cv2.imwrite(
            "G:/Project dataset/Manoj/Mnist ready/" + str(x) + "/" + str(c) + ".jpg",
            pix_img,
        )
    except:
        print("hello")
    return img


def txt_file(x, y):
    f = open("G:/Project/errors.txt", "r")
    f_contents = f.read().split("\n")
    for fc in f_contents:
        f_num = fc.split("-")  # seperates xy and points
        if f_num[0] == str(x) + str(y):
            f_xy = f_num[1].split(" ")  # gives us x,y list of strings
    # print(f_xy)
    xy_points = []
    for xy_index in range(0, len(f_xy)):
        xy_seperated = f_xy[xy_index].split(",")
        xy_points.append([int(xy_seperated[0]) // 100, int(xy_seperated[1]) // 100])
    f.close()
    return xy_points


def main():
    for x in range(6, 7):  # 1,13
        c = 0
        ec = 1  # ec = error count
        for y in range(1, 7):  # 1,7
            image = cv2.imread(
                "G:/Project dataset/Manoj/mine/" + str(x) + str(y) + ".jpg"
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # count = 0
            xp = txt_file(x, y)

            for i in range(0, 15):
                for j in range(0, 24):
                    # traversing through each element
                    # means that this i j value has error in it
                    cut_img = image[j * 100 : j * 100 + 130, i * 100 : i * 100 + 130]
                    cv2.imshow("f", cut_img)
                    if [i, j] in xp:  # True if points i,j are erroraneous
                        cv2.imwrite(
                            "G:/Project dataset/Manoj/set_of_Errors/"
                            + str(x)
                            + "_"
                            + str(ec)
                            + ".jpg",
                            cut_img,
                        )
                        ec += 1
                        for zero_i in range(0, 100):
                            for zero_j in range(0, 100):
                                image[j * 100 + zero_i][i * 100 + zero_j] = 0
                    else:
                        c += 1
                        image = cut_img_contour(image, x, y, cut_img, c)
                    # count+=1
                    # if i j are not erroranous
            # print(xp)
            # print(count,len(xp))
            # image,cc = crop(image)
            # print(x,y,cc-1,ec-1,cc+ec-2)
            print(x, y)
        print(ec - 1)
        print(c)
    # cv2.imshow('Image',cv2.resize(image,(0,0),fx=0.33,fy=0.33))
    return 0


main()
cv2.waitKey(0)
cv2.destroyAllWindows()
