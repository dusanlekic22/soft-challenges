# import libraries here
import numpy as np
import cv2 # OpenCV biblioteka


def hist(image):
    height, width = image.shape[0:2]
    x = range(0, 256)
    y = np.zeros(256)

    for i in range(0, height):
        for j in range(0, width):
            pixel = image[i, j]
            y[pixel] += 1

    return (x, y)


def count_cars(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj prebrojanih automobila. Koristiti ovu putanju koja vec dolazi
    kroz argument procedure i ne hardkodirati nove putanje u kodu.

    Ova procedura se poziva automatski iz main procedure i taj deo koda nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih automobila
    """
    car_count = 0
    # TODO - Prebrojati auta i vratiti njihov broj kao povratnu vrednost ove procedure
    img = cv2.imread(image_path)  # ucitavanje slike sa diska
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]


    if imgHeight > imgWidth:
        img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)

    img =cv2.resize(img, (1000,700), interpolation = cv2.INTER_AREA)
    # plt.imshow(img, 'gray')
    # plt.show()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img_gray,'gray')
    # plt.show()

    blurred = cv2.GaussianBlur(img_gray, (3, 3), cv2.BORDER_DEFAULT)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_gray = cv2.erode(img_gray, kernel)


    # ret, image_bin = cv2.threshold(img_gray, 0, 255,
    #                                 cv2.THRESH_OTSU)  # ret je izracunata vrednost praga, image_bin je binarna slika
    # print("Otsu's threshold: " + str(ret))
    # plt.imshow( image_bin,'gray')
    # plt.show()
    if imgWidth < 500 and imgHeight < 500:
        image_bin = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 43,30)
    else:
        #image_bin = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 20)
        ret, image_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

    image_bin = 255 - image_bin
    # ret, image_bin = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY);
    # plt.imshow(image_bin,'gray')
    # plt.show()


    # plt.imshow(image_bin, 'gray')
    # plt.show()

    cv2.imshow("Binary", np.hstack((blurred, image_bin)))
    #cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()

    open = image_bin.copy()

    if imgWidth>500 and imgHeight>500:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
        open = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, kernel)

    cv2.imshow("Opening", np.hstack((image_bin, open)))
    #cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()

    dilation = open.copy()

    if imgWidth < 500 and imgHeight < 500:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        dilation = cv2.dilate(open, kernel, iterations=1)



    cv2.imshow("Dilate", np.hstack(( open, dilation)))
    #cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()


    imgc, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #RETR_EXTERNAL I HIERARCHY

    #print(len(contours))

    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    cv2.imshow("Countours1", img)
    #cv2.waitKey(0)  # waits until a key is pressed000
    cv2.destroyAllWindows()
    small_contours = 0
    big_contours = 0
    contours_car = []  # ovde ce biti samo konture koje pripadaju bar-kodu

    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        # if -100 <2.6*width -height< 100 or -100 <2.6*width -height< 100:  # uslov da kontura pripada bar-kodu
        if  width>30 and height >30 and width<70 and height <70:
            small_contours+=1
        elif width>70 and height >70:
            big_contours+=1

    print ("Big countours:", big_contours)
    print("Small countours:", small_contours)
    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        # if -100 <2.6*width -height< 100 or -100 <2.6*width -height< 100:  # uslov da kontura pripada bar-kodu
        if  imgWidth<500 and imgHeight<500 :
            if  width>40 and height >40 and height<150 and width <150:
                contours_car.append(contour)  # ova kontura pripada kolima
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            print("Mali automobili")
        elif imgWidth>500 and imgHeight>500:
            if (abs(2*width-height)<75 or abs(2*height-width)<75) and  width > 70 and height > 70 :
                contours_car.append(contour)  # ova kontura pripada kolima
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            print("Veliki")
        else:
            if  width>40 and height >40:
                contours_car.append(contour)  # ova kontura pripada kolima
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (0, 0, 255), 2)


    cv2.imshow("Countours2", img)
    #cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()
    cv2.drawContours(img, contours_car, -1, (0, 255, 0), 1)
    # cv2.imshow("Countours2", img)
    # #cv2.waitKey(0)  # waits until a key is pressed
    car_count = len(contours_car)
    print("Cars:",car_count)

    return car_count
