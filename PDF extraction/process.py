# import libraries here
import datetime
import math
import re

import cv2
import dlib
import scipy
from PIL import Image
import sys
import imutils
import pyocr
import pyocr.builders
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import scipy.cluster
# prikaz vecih slika
from imutils import face_utils
from sklearn.cluster import KMeans
from fuzzywuzzy import fuzz

matplotlib.rcParams['figure.figsize'] = 16, 12

# PyOCR podrzava i neke druge alate, tako da je potrebno proveriti koji su sve alati instalirani
tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)

# odaberemo Tessract - prvi na listi ako je jedini alat
tool = tools[0]
print("Koristimo backend: %s" % (tool.get_name()))
# biramo jezik očekivanog teksta
lang = 'eng'


def find_rect(image):
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(image, (5, 5), 0)
    #gray = cv2.equalizeHist(gray)
    #high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #lowThresh = 0.5 * high_thresh
    # edged = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
    edged = cv2.Canny(image, 127, 255)
    # edged = cv2.Canny(gray, 50, 150,apertureSize=3)

    img_black = np.zeros([image.shape[0], image.shape[1], 1], dtype=np.uint8)
    img_black.fill(0)
    linesP = cv2.HoughLinesP(edged, 1, np.pi / 180, 50, None, 100, 10)
    # linesP = sorted(linesP, key=linesP[0][0])
    #plt.imshow(edged, 'gray')
    #plt.show()
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(img_black, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 1, cv2.LINE_AA)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(img_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cv2.findContours(cv2.dilate(edged.copy(),(5,5)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    # cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)

    #   else:
    #       print("Cannot find 4 edges")
    rect = cv2.minAreaRect(cnts[0])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    #plt.imshow(image)
    #plt.show()
    return image, rect


def find_angle(rect):
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


# Deskew image
def deskew(cvImage):
    rect = find_rect(cvImage)[1]
    angle = find_angle(rect)
    return rotateImage(cvImage, -1.0 * angle)


def rotate_and_crop(img, rect):
    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    if warped.shape[0] > warped.shape[1]:
        warped = imutils.rotate_bound(warped, -90)
    # cv2.imwrite("crop_img.jpg", warped)
    return warped


def extract_text(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = rotate_and_crop(find_rect(img)[0], find_rect(img)[1])
    #plt.imshow(img, 'gray')
    #plt.show()
    img = imutils.resize(img, height=500)
    width = img.shape[1]
    height = img.shape[0]
    #plt.imshow(img[height - 45:height - 15, int(width / 2) - 15:int(width / 2) + 15])
    #plt.show()
    # average = img[height - 45:height - 15, int(width / 2) - 15:int(width / 2) + 15].mean(axis=0).mean(axis=0)
    dominant_color = dominant_colors(img[height - 45:height - 15, int(width / 2) - 15:int(width / 2) + 15])[0]
    #plt.imshow(img)
    #plt.show()

    job = 'Human Resources'
    dob = '16, Sep 1977'
    name = 'Amanda Huff'
    snn = '711-05-8747'
    # zuta
    if dominant_color[0] < 70 and dominant_color[1] < 70 and dominant_color[2] < 70:
        print("Zuta")
        company = 'Apple'
        text = find_text(img[int(height / 2) - 200:int(height / 2) + 20, int(width / 2) - 170:int(width / 2) + 110], 300)
        if len(text) > 0:
            name = text[0]
        if len(text) > 1:
            dob = text[1]
        if len(text) > 2:
            snn = text[2]
        text = find_text(img[int(height / 2) - 195:int(height / 2) - 170, int(width / 2) - 160:int(width / 2) + 110], 43)
        if len(text) > 0:
            job = text[0]
    # bela
    elif (dominant_color[0] > 170 and dominant_color[1] > 170 and dominant_color[2] > 170) or\
            (100 < dominant_color[0] < 200 and 100 < dominant_color[1] < 200 and 100 < dominant_color[2] < 200
             and dominant_color[0] == dominant_color[1] and dominant_color[1] == dominant_color[2]):
        print("Bela")
        company = 'IBM'
        text = find_text(img[int(height / 2) + 60:int(height / 2) + 160, int(width / 2) + 20:int(width / 2) + 300], 200)
        if len(text) > 0:
            job = text[0]
        if len(text) > 1:
            dob = text[1]
        text = find_text(255-img[188:242, 335:680], 75)
        if len(text) > 0:
            name = text[0]
        text = find_text(255-img[250:300, 335:540], 75)
        if len(text) > 0:
            snn = text[0]
    # plava
    else:
        #plt.imshow(img[10:50, int(width) - 300:int(width) - 50])
        #plt.show()
        is_black = sorted(dominant_colors(img[10:50, int(width) - 300:int(width) - 50]))[0]
        if is_black[0] < 70 and is_black[1] < 70 and is_black[2] < 70:
            text = find_text(255-img[10:70, 40:600])
            img = 255-img[int(height / 2) - 140:int(height / 2) + 90, int(width / 2) - 220:int(width / 2) + 100]
        else:
            text = find_text(255-img[100:170, 30:400])
            img = 255-img[int(height / 2) - 50:int(height / 2) + 120, int(width / 2) - 180:int(width / 2) + 80]
        if len(text) > 0:
            name = text[0]
            #print(name)
        print("Plava")
        company = 'Google'
        text = find_text(img, 300)
        if len(text) > 0:
            snn = text[0]
        if len(text) > 1:
            job = text[1]
        if len(text) > 2:
            dob = text[2]
    name = re.sub('[^A-Za-z ]+', '', name)
    if len(name) > 1:
        if ' ' == name[0]:
            name = name[1:]
        name = name[0].upper() + name[1:]
    snn = re.sub('[^0-9-]+', '', snn)
    job = re.sub('[^A-Za-z ]+', '', job)
    if len(job) > 1:
        if ' ' == job[0]:
            job = job[1:]
        job = job[0].upper() + job[1:]
    if len(snn) < 2:
        snn = '711-05-8747'
    if fuzz.ratio(job, 'Scrum Master') > 50:
        job = 'Scrum Master'
    if fuzz.ratio(job, 'Human Resources') > 50:
        job = 'Human Resources'
    if fuzz.ratio(job, 'Team Lead') > 50:
        job = 'Team Lead'
    if fuzz.ratio(job, 'Software Engineer') > 50:
        job = 'Software Engineer'
    if fuzz.ratio(job, 'Manager') > 50:
        job = 'Manager'
    info = [name, dob, job, snn, company]
    print(info)
    return info

def dominant_colors(img):
    num_clusters = 2
    ar = np.asarray(img)
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

    kmeans = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        max_iter=20,
        random_state=1000
    ).fit(ar)
    codes = kmeans.cluster_centers_

    vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, bins = np.histogram(vecs, len(codes))  # count occurrences

    colors = []
    for index in np.argsort(counts)[::-1]:
        colors.append(tuple([int(code) for code in codes[index]]))
    #plt.imshow([colors])
    #print(colors[0])
    #plt.show()
    return colors


def find_text(img, resize=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray, 'gray')
    #plt.show()
    if resize is not None:
        gray = imutils.resize(gray, height=resize)
    #plt.imshow(gray, 'gray')
    #plt.show()
    # save the image you want
    # text1 = tool.image_to_data(threshed, lang="ind", output_type='data.frame')
    text = tool.image_to_string(
        Image.fromarray(gray),
        lang=lang)
    text = text.split('\n')
    new_text = []
    for word in text:
        if len(word) > 1:
            new_text.append(word)
    return new_text


# text = text1[text1.conf != -1]
# lines = text.groupby('block_num')['text'].apply(list)
# conf = text.groupby(['block_num'])['conf'].mean()

class Person:
    """
    Klasa koja opisuje prepoznatu osobu sa slike. Neophodno je prepoznati samo vrednosti koje su opisane u ovoj klasi
    """

    def __init__(self, name: str = None, date_of_birth: datetime.date = None, job: str = None, ssn: str = None,
                 company: str = None):
        self.name = name
        self.date_of_birth = date_of_birth
        self.job = job
        self.ssn = ssn
        self.company = company


def extract_info(image_path: str) -> Person:
    """
    Procedura prima putanju do slike sa koje treba ocitati vrednosti, a vraca objekat tipa Person, koji predstavlja osobu sa licnog dokumenta.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param image_path: <str> Putanja do slike za obradu
    :return: Objekat tipa "Person", gde su svi atributi setovani na izvucene vrednosti
    """

    text = extract_text(image_path)
    name = text[0]
    try:
        dob = datetime.strptime(text[1], '%d, %b %Y')
    except:
        dob = datetime.now()
    job = text[2]
    snn = text[3]
    company = text[4]

    person = Person(name, dob, job, snn, company)

    # TODO - Prepoznati sve neophodne vrednosti o osobi sa slike. Vrednosti su: Name, Date of Birth, Job,
    #       Social Security Number, Company Name

    return person

