import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow_datasets.public_api as tfds
from tensorflow import transpose
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def show_image(image, value=None):
    """
    This function plot the image in matplotlib to allow preview of image
    :param image: the image
    :param value: the label of the image (Default: None)
    :return: None
    """
    plt.imshow(image, cmap="gray")
    if value is not None:
        plt.title(f"Value: {value}")
    plt.show()

def plotTrainingResult(data):
    """
    This function plot the training result in each epoch (Accuracy and loss)
    :param data: the history data
    :return: None
    """
    plt.figure(figsize=[7, 5])
    accuracy = data['accuracy']
    test_accuracy = data['val_accuracy']
    loss = data['loss']
    test_loss = data['val_loss']
    epochs = range(len(accuracy))

    plt.subplot(121)
    plt.plot(epochs, accuracy, 'bo', label="Training accuracy")
    plt.plot(epochs, test_accuracy, 'b', label="Testing accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(122)
    plt.plot(epochs, loss, 'bo', label="Training loss")
    plt.plot(epochs, test_loss, 'b', label="Testing loss")
    plt.title("Loss")
    plt.legend()
    plt.show()


def load_dataset():
    """
    This function load mnist dataset from keras library
    """

    source = tfds.load("emnist", split="train", as_supervised=True)
    train_X = []
    train_y = []
    for image, label in source:
        train_X.append(transpose(image[:, :, 0]))
        train_y.append(label)

    """
    test_X = []
    test_y = []
    source = tfds.load("emnist", split="test", as_supervised=True)
    for image, label in source:
        test_X.append(transpose(image[:, :, 0]))
        test_y.append(label)
    """

    data = np.vstack([train_X])
    labels = np.hstack([train_y])

    return data, labels

def preprocess_data():
    """
    Preprocess data before perform training including:
    resize image
    scaling image pixels from [0,255] to [0,1]
    split 80% training and 20 %testing data

    :return: (x_train,x_test,y_train,y_test): train test splited data, num_classes: number of output class to be predicted
    """
    (data, labels) = load_dataset()
    print("Data loaded, proceed to preprocessing")
    #data = [cv2.resize(image, (32, 32)) for image in data]
    data = np.array(data, dtype="float32")
    data = np.expand_dims(data, axis=-1)
    data /= 255.0

    print("Done preproccessing")


    # inspect size of dataset
    print("data shape :", data.shape, labels.shape)

    # find unique numbers from train labels
    classes = np.unique(labels)
    num_class = len(classes)
    print("Total number of outputs: ", num_class)
    print("Output classes:", classes)

    encoded_labels = to_categorical(labels)

    (x_train, x_test, y_train, y_test) = train_test_split(data, encoded_labels, test_size=0.2, stratify=labels,
                                                          random_state=0)

    return (x_train, x_test, y_train, y_test), num_class


def build_cnn(num_class):
    """
    Function to build the CNN network

    :param num_class: number of output class to be predicted
    :return: model: the compiled cnn network model
    """
    # create the model
    model = Sequential()

    # add model layers
    model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(num_class, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model(file):
    """
    The whole process to train the model from preprocessing until save the trained model
    :return: None
    """
    # data preprocessing
    (X_train, X_valid, Y_train, Y_valid), num_class = preprocess_data()

    # build CNN network
    # parameters
    batch_size = 32
    epochs = 16

    cnn = build_cnn(num_class)

    # train model
    train_results = cnn.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                              validation_data=(X_valid, Y_valid))

    # plot training results in each epoch
    plotTrainingResult(train_results.history)

    cnn.save(file)
    return cnn

def sort_contours(cnts):
    """
    Sort the contours from left to right and top to bottom
    :param cnts: list of contours
    :return: cnts: sorted contour, bounding_boxes: the position of contour
    """
    # get the list of bounding boxes
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    # sort from left to right, top to bottom
    cnts, bounding_boxes = zip(*sorted(zip(cnts, bounding_boxes), key=lambda b: b[1][0]))
    return cnts, bounding_boxes

def resize(thresh):
    """
    Resize the image to defined size
    :param thresh: the image to be resized
    :return: resized: the resized image
    """
    max_size = 28
    (tW, tH) = thresh.shape
    dim = None
    (h, w) = thresh.shape[:2]

    if tW > tH:
        r = max_size / float(w)
        dim = (max_size, int(h * r))
    else:
        r = max_size / float(h)
        dim = (int(w * r), max_size)

    resized = cv2.resize(thresh, dim, interpolation=cv2.INTER_AREA)
    return resized


def predict_image(image, model):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_list = contours[0]
    contours_list = sort_contours(contours_list)[0]

    chars = []

    for c in contours_list:
        (x,y,w,h) = cv2.boundingRect(c)
        print(x,y,w,h)
        if (30 <= w <= 200) and (30 <= h <=200):
            # extract character and threshold to let character appear as white on back background
            # grab width and height of thresholded image
            roi = blurred[y:y+h, x:x+w]
            threshold = cv2.threshold(roi, 30, 150, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            resized_image = resize(threshold)

            # get dimention of image
            # calculate how much to be padding
            (tH, tW) = resized_image.shape
            dX = int(max(20, 28-tW)/2.0)
            dY = int(max(20, 28-tH)/2.0)

            padded = cv2.copyMakeBorder(resized_image, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
            padded = cv2.resize(padded, (28,28))

            # prepare padded image for classification
            padded = padded.astype("float32")/255.0
            padded = np.expand_dims(padded, axis=-1)

            chars.append((padded,(x,y,w,h)))

    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars],dtype="float32")

    if len(chars) > 0:
        preds = model.predict(chars)

        labelNames = [i for i in range(10)]

        for x in range(ord('A'), ord('Z') + 1):
            labelNames.append(chr(x))
        for x in range(ord('a'), ord('z') + 1):
            labelNames.append(chr(x))
        recognized_character = []

        for(pred, (x,y,w,h)) in zip(preds, boxes):
            i = np.argmax(pred)
            prob = pred[i]
            label = labelNames[i]

            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(image, f"{label}, {prob:.2f}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            recognized_character.append(label)

        print("Character recognized: ", recognized_character)
    return image


if __name__ == "__main__":
    mode = "image"
    file = "./OCR/model.h5"
    image_file = "./OCR/sample_image/image.jpg"
    model = None

    if not os.path.exists(file):
        model = train_model(file)
    else:
        model = load_model(file)

    if mode == "image":
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        annotated_image = predict_image(image, model)
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    elif mode == "webcam":
        cam_port = 0
        video_capture = cv2.VideoCapture(cam_port)
        win_name = "Camera Mode"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        while cv2.waitKey(1) != 27:
            has_frame, frame = video_capture.read()
            frame_flip = cv2.flip(frame, 1)
            if not has_frame:
                break
            annotated_image = predict_image(frame_flip, model)
            cv2.imshow(win_name, annotated_image)

        video_capture.release()
        cv2.destroyWindow(win_name)