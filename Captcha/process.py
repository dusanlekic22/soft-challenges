from __future__ import print_function
# import libraries here
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

# keras
import keras.backend.theano_backend
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD
from keras.models import model_from_json

#Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = 16, 12


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def resize_text(first,other):
    resized = other
    if other.shape[1]<first.shape[1]:
        scale_percent = 100 * other.shape[1]/first.shape[1]
        width = int(other.shape[1] * scale_percent / 100)
        height = int(other.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(other, dim, interpolation=cv2.INTER_AREA)
    return resized
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    #image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
    ret, image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_OTSU)
    return image_bin
def invert(image):
    return 255-image
def opening(image_bin,dim):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (dim, dim))
    open = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, kernel)
    return open
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized
def scale_to_range(image):
    return image / 255
def matrix_to_vector(image):
    return image.flatten()
def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann
def convert_output(outputs):
    return np.eye(len(outputs))
def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def merge_regions(regions_array,image_bin):
    #regions_array = sorted(regions, key=lambda item: item[1][1])
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    merged_regions = []
    skip=False
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(regions_array)):
        #Preskace iteracije gde se ubacuju kukice
        if skip:
            skip=False
            continue
        if index==len(regions_array)-1:
            merged_regions.append(regions_array[index])
            break
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = current[1] - (next_rect[1] + next_rect[3])  # Y_next - (Y_current + Y_current)
        #display_image(regions_array[index][0])
        if 5<current[0]+current[2]/2-next_rect[0]+next_rect[2]/2 and current[0]+current[2]/2-next_rect[0]+next_rect[2]/2<100 and current[2]>next_rect[2]:
            x=current[0]
            y=next_rect[1]
            w=current[2]
            h=current[3]+next_rect[3]
            #print(x,y,w,h)
            region = image_bin[y:y + h + 1, x:x + w + 1]
            #display_image(region)
            rect=(x,y,w,h)
            merged_regions.append((resize_region(region),rect))
            skip=True
        else:
            merged_regions.append(regions_array[index])
            skip = False
            #display_image(regions_array[index][0])
    return merged_regions

def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    #img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img,contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    #cv2.drawContours(cvImage, contours, -1, (255, 0, 0), 1)
    #display_image(cvImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
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
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

def select_roi(image_orig, image_bin):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    regions_array = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        region = image_bin[y:y+h+1,x:x+w+1]
        if h > 15 and w > 15:
            regions_array.append([resize_region(region), (x,y,w,h)])
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)


    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    regions_array = merge_regions(regions_array,image_bin)

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles)-1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index+1]
        distance = next_rect[0] - (current[0]+current[2]) #X_next - (X_current + W_current)
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances


def create_ann():
    '''
    Implementirati veštačku neuronsku mrežu sa 28x28 ulaznih neurona i jednim skrivenim slojem od 128 neurona.
    Odrediti broj izlaznih neurona. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    # Postaviti slojeve neurona mreže 'ann'
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(60, activation='sigmoid'))

    return ann


def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=3000, batch_size=1, verbose=0, shuffle=False)

    return ann


def serialize_ann(ann):
    # serijalizuj arhitekturu neuronske mreze u JSON fajl
    model_json = ann.to_json()
    with open("serialization_folder/neuronska.json", "w") as json_file:
        json_file.write(model_json)
    # serijalizuj tezine u HDF5 fajl
    ann.save_weights("serialization_folder/neuronska.h5")


def load_trained_ann():
    try:
        # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
        json_file = open('serialization_folder/neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        # ucitaj tezine u prethodno kreirani model
        print(ann.load_weights("serialization_folder/neuronska.h5"))
        print("Istrenirani model uspesno ucitan.")
        return ann
    except Exception as e:
        # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        return None
def load_train_img(img):
    image_color = load_image(img)
    #if 'dataset' in img:
    #    first= load_image('dataset/validation/train0.bmp')
    #    image_color=resize_text(first,load_image(img))
    #    display_image(image_color)
    image_color = deskew(image_color)
    img_bin = image_bin(image_gray(image_color))
    inv = invert(img_bin)
    #inv = opening(inv,5)
    selected_regions, letters, region_distances = select_roi(image_color.copy(), inv)
    #display_image(selected_regions)
    inputs = prepare_for_ann(letters)
    print('Broj prepoznatih regiona:', len(letters))
    return inputs

def train_or_load_character_recognition_model(train_image_paths):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta)

    Procedura treba da istrenira model i da ga sacuva pod proizvoljnim nazivom. Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran

    :param train_image_paths: putanje do fotografija alfabeta
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati ako je vec istreniran
    model = None

    inputs_major = load_train_img(train_image_paths[0])
    inputs_minor = load_train_img(train_image_paths[1])
    inputs = inputs_major + inputs_minor
    #print(inputs)
    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'Š', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u',
                'v', 'w', 'x', 'y', 'z', 'ž']
#
    output = convert_output(alphabet)

    # probaj da ucitas prethodno istreniran model
    model = load_trained_ann()
#
    # ako je ann=None, znaci da model nije ucitan u prethodnoj metodi i da je potrebno istrenirati novu mrezu
    if model == None:
        print("Treniranje modela zapoceto.")
        model = create_ann()
        model = train_ann(model, inputs, output)
        print("Treniranje modela zavrseno.")
        # serijalizuj novu mrezu nakon treniranja, da se ne trenira ponovo svaki put
        serialize_ann(model)


    return model

def kmeans_words(image_path):
    image_color = load_image(image_path)
    img = image_bin(image_gray(image_color))
    inv = invert(img)
    selected_regions, letters, distances = select_roi(image_color.copy(), inv)
    #print(distances)
    # Podešavanje centara grupa K-means algoritmom
    distances = np.array(distances).reshape(len(distances), 1)
    # Neophodno je da u K-means algoritam bude prosleđena matrica u kojoj vrste određuju elemente
    if len(distances) < 3 :
        return None
    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
    k_means.fit(distances)
    return k_means

def display_result(outputs, alphabet, k_means):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    # Odrediti indeks grupe koja odgovara rastojanju između reči, pomoću vrednosti iz k_means.cluster_centers_
    w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]
        # Dodati space karakter u slučaju da odgovarajuće rastojanje između dva slova odgovara razmaku između reči.
        # U ovu svrhu, koristiti atribut niz k_means.labels_ koji sadrži sortirana rastojanja između susednih slova.
        if idx==len(k_means.labels_):
            break
        if (k_means.labels_[idx] == w_space_group):
            result += ' '
        result += alphabet[winner(output)]
    print(result)
    return result

def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """
    extracted_text = ""

    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                'Š', 'T', 'U','V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                'k', 'l', 'm','n', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u','v', 'w', 'x', 'y', 'z', 'ž']
    inputs = load_train_img(image_path)
    results = trained_model.predict(np.array(inputs, np.float32))
    if kmeans_words(image_path)==None:
        return extracted_text
    extracted_text = display_result(results, alphabet, kmeans_words(image_path))
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string

    return extracted_text

