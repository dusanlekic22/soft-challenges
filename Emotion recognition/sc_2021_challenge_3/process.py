# import libraries here
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC  # SVM klasifikator
from joblib import dump, load
from imutils.face_utils import FaceAligner

emotions = ['anger', 'contempt', 'disgust', 'happiness', 'neutral', 'sadness', 'surprise']


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def dominant_features(image):
    # inicijalizaclija dlib detektora (HOG)
    detector = dlib.get_frontal_face_detector()
    # ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # ucitavanje i transformacija slike
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detekcija svih lica na grayscale slici
    rects = detector(gray, 1)
    circles = []
    # iteriramo kroz sve detekcije korak 1.
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        # odredjivanje kljucnih tacaka - korak 2
        shape = predictor(gray, rect)
        # shape predstavlja 68 koordinata
        shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz
        # print("Dimenzije prediktor matrice: {0}".format(shape.shape))  # 68 tacaka (x,y)
        # print("Prva 3 elementa matrice")
        # print(shape[:3])
        circles.append(shape)
        # konvertovanje pravougaonika u bounding box koorinate
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # crtanje pravougaonika oko detektovanog lica
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #
        ## ispis rednog broja detektovanog lica
        # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # crtanje kljucnih tacaka
        # for (a, b) in shape:
        #    cv2.circle(gray, (a, b), 1, (0, 0, 0),10)

        image = gray[y:y + h, x:x + w]
        # plt.imshow(image,'gray')
        # plt.show()
    return image


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx * ny))


def face_remap(shape):
    remapped_image = cv2.convexHull(shape)
    return remapped_image


# source: https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

def align_face(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # show the original input image and detect faces in the grayscale
    # image
    # cv2.imshow("Input", gray)
    # face_crop = False
    rects = detector(gray, 0)
    for rect in rects:
        # print('Djokovic')
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        face_aligned = fa.align(gray, gray, rect)

        # rects3 = detector(face_aligned, 1)
        # if len(rects3) > 0:
        #    shape = predictor(face_aligned, rects3[0])
        #    # plt.imshow(shape, 'gray')
        #    # plt.show()
        #    shape = face_utils.shape_to_np(shape)
        #    out_face = np.zeros_like(face_aligned)
        #    # initialize mask array
        #    remapped_shape = np.zeros_like(shape)
        #    feature_mask = np.zeros((face_aligned.shape[0], face_aligned.shape[1]))
        #
        #    # we extract the face
        #    remapped_shape = face_remap(shape)
        #    cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
        #    feature_mask = feature_mask.astype(np.bool)
        #    out_face[feature_mask] = face_aligned[feature_mask]
        #    face_crop = True

        rects2 = detector(face_aligned, 1)
        # display the output images
        # cv2.imshow("Original", faceOrig)
        # cv2.imshow("Aligned", face_aligned)
        # cv2.waitKey(0)

        if len(rects2) > 0:
            # print('Drugi')
            (x2, y2, w2, h2) = face_utils.rect_to_bb(rects2[0])
            crop = face_aligned[y2:y2 + h2, x2:x2 + w2]
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                face_aligned = crop

        face_aligned = imutils.resize(face_aligned, 200)
        # face_aligned = cv2.resize(face_aligned, (100, 100))
        # plt.imshow(face_aligned, 'gray')
        # plt.show()
        return face_aligned


def train_or_load_facial_expression_recognition_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje i listu labela za svaku fotografiju iz prethodne liste

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno istreniran. 
    Ako serijalizujete model, serijalizujte ga odmah pored main.py, bez kreiranja dodatnih foldera.
    Napomena: Platforma ne vrsi download serijalizovanih modela i bilo kakvih foldera i sve ce se na njoj ponovo trenirati (vodite racuna o vremenu). 
    Serijalizaciju mozete raditi samo da ubrzate razvoj lokalno, da se model ne trenira svaki put.

    Vreme izvrsavanja celog resenja je ograniceno na maksimalno 1h.

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran
    try:
        return (load('svm.joblib'),load('svm_bin.joblib'))
    except Exception as e:
        # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        model = None

    anger_imgs = []
    contempt_imgs = []
    disgust_imgs = []
    happiness_imgs = []
    neutral_imgs = []
    sadness_imgs = []
    surprise_imgs = []
    anger_or_disgust_imgs = []
    # load_images()
    labels = []
    labels_bin = []
    ang = cont = disg = hap = neu = sad = sur =0
    for img_path, img_label in zip(train_image_paths, train_image_labels):
        # align_face('./dataset/validation/image-81.jpg')
        if 'anger' in img_label:
            ang+=1
        elif 'contempt' in img_label:
            cont+=1
        elif 'disgust' in img_label:
            disg += 1
        elif 'happiness' in img_label:
            hap += 1
        elif 'neutral' in img_label:
            neu+=1
        elif 'sadness' in img_label:
            sad+=1
        elif 'surprise' in img_label:
            sur+=1
    max_number = max([ang, cont, disg, hap, neu, sad, sur])
    #print(max_number)

    for img_path, img_label in zip(train_image_paths, train_image_labels):
        # align_face('./dataset/validation/image-81.jpg')
        #print(img_path)
        img = align_face(img_path)
        # img = prepare_image(img_path)
        # inputs.append(hog.compute(img))
        # if img.shape[0]!=762 and img.shape[1]!=562:
        #    print(img.shape, img_path)
        if 'anger' in img_label:
            anger_imgs.append(img)
            anger_or_disgust_imgs.append(img)
            if max_number - ang > 0:
                anger_imgs.append(img)
                anger_or_disgust_imgs.append(img)
                ang+=1
        elif 'contempt' in img_label:
            contempt_imgs.append(img)
            if max_number - cont > 0:
                contempt_imgs.append(img)
                cont+=1
        elif 'disgust' in img_label:
            disgust_imgs.append(img)
            anger_or_disgust_imgs.append(img)
            if max_number - disg > 0:
                disgust_imgs.append(img)
                anger_or_disgust_imgs.append(img)
                disg += 1
        elif 'happiness' in img_label:
            happiness_imgs.append(img)
            if max_number - hap > 0:
                happiness_imgs.append(img)
                hap+=1
        elif 'neutral' in img_label:
            neutral_imgs.append(img)
            if max_number - neu > 0:
                neutral_imgs.append(img)
                neu+=1
        elif 'sadness' in img_label:
            sadness_imgs.append(img)
            if max_number - sad > 0:
                sadness_imgs.append(img)
                sad+=1
        elif 'surprise' in img_label:
            surprise_imgs.append(img)
            if max_number - sur > 0:
                surprise_imgs.append(img)
                sur+=1

    # print(len(anger_imgs),len(disgust_imgs),len(anger_or_disgust_imgs))
    hog = start_hog(img)

    anger_features = []
    contempt_features = []
    disgust_features = []
    happiness_features = []
    neutral_features = []
    sadness_features = []
    surprise_features = []
    anger_or_disgust_features = []

    for img in anger_imgs:
        anger_features.append(hog.compute(img))
        labels_bin.append('anger')

    for img in disgust_imgs:
        disgust_features.append(hog.compute(img))
        labels_bin.append('disgust')

    for img in contempt_imgs:
        contempt_features.append(hog.compute(img))
        labels.append('contempt')

    for img in happiness_imgs:
        happiness_features.append(hog.compute(img))
        labels.append('happiness')

    for img in neutral_imgs:
        neutral_features.append(hog.compute(img))
        labels.append('neutral')

    for img in sadness_imgs:
        sadness_features.append(hog.compute(img))
        labels.append('sadness')

    for img in surprise_imgs:
        surprise_features.append(hog.compute(img))
        labels.append('surprise')

    for img in anger_or_disgust_imgs:
        anger_or_disgust_features.append(hog.compute(img))
        labels.append('anger_or_disgust')

    anger_features = np.array(anger_features)
    contempt_features = np.array(contempt_features)
    disgust_features = np.array(disgust_features)
    happiness_features = np.array(happiness_features)
    neutral_features = np.array(neutral_features)
    sadness_features = np.array(sadness_features)
    surprise_features = np.array(surprise_features)
    anger_or_disgust_features = np.array(anger_or_disgust_features)

    x = np.vstack((contempt_features, happiness_features, neutral_features,
                   sadness_features, surprise_features, anger_or_disgust_features))

    y = np.array(labels)

    x_bin = np.vstack((anger_features, disgust_features))
    y_bin = np.array(labels_bin)
    # print(x.shape)
    # clf_svm = SVC(kernel='linear', probability=True)
    # clf_svm.fit(inputs, train_image_labels)

    x = reshape_data(x)
    x_bin = reshape_data(x_bin)
    # print(x.shape)
    clf_svm = SVC(kernel='linear', probability=True)
    clf_svm.fit(x, y)
    dump(clf_svm, 'svm.joblib')

    clf_svm_bin = SVC(kernel='linear', probability=True)
    clf_svm_bin.fit(x_bin, y_bin)
    dump(clf_svm_bin, 'svm_bin.joblib')
    # y_train_pred = clf_svm.predict(x)
    # print("Train accuracy: ", accuracy_score(labels, y_train_pred))

    model = (clf_svm, clf_svm_bin)

    return model


def prepare_image(img_path):
    img = cv2.imread(img_path)
    # img = load_image(img_path)
    img = cv2.resize(img, (562, 762), interpolation=cv2.INTER_AREA)
    # if 'img9' in img_path:
    #    plt.imshow(img)
    #    plt.show()
    return img


def start_hog(img):
    nbins = 9  # broj binova
    cell_size = (10, 10)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    return hog


def reshape_test(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples * nx, ny))


def extract_facial_expression_from_image(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje ekspresije lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati ekspresiju.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <String>  Naziv prediktovane klase (moguce vrednosti su: 'anger', 'contempt', 'disgust', 'happiness', 'neutral', 'sadness', 'surprise'
    """

    # print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
    # print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)
    # if 'image-90.jpg' in image_path:
    #    dominant_features(image_path)

    img = align_face(image_path)
    # img = prepare_image(image_path)
    # print(image_path)

    hog = start_hog(img)

    hog_image = hog.compute(img)
    test_data = reshape_data(np.array([hog_image]))
    # print(reshape_data(np.array([hog_image])).shape)
    output = trained_model[0].predict(test_data)[0]
    bin_output = trained_model[1].predict(test_data)[0]
    if 'anger_or_disgust' in output:
        #print('anger or disgust')
        facial_expression = bin_output
    else:
        facial_expression = output
    # print(facial_expression)
    return facial_expression
# dominant_features('dataset/train/image-714.jpg')

# align_face('dataset/validation/image-82.jpg')
