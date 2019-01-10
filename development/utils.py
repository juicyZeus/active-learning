import numpy as np
import cv2
import os


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    
    img = cv2.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
   
    # resize to 224, 224
    resized_img = cv2.resize(img,(224, 224), interpolation = cv2.INTER_CUBIC)

    return resized_img


# load data from folder
def load_data_folder(data_dir):
    
    x = []
    y = []
    
    # get file names
    # remove irrelevant file names
    file_names = [fname for fname in os.listdir(data_dir) if "x" in fname]
    
    # load training data
    for file_name in file_names:

        img_folder = data_dir + file_name 
        img_paths = os.listdir(img_folder)
        img_paths = [img_name for img_name in img_paths if img_name.endswith("png")]

        for img_path in img_paths:
            y.append(file_name)

            img_full_path = img_folder + "/" + img_path
            img = load_image(img_full_path)
            x.append(img)
    
    
    x_array = np.array(x)
    y_array = np.array(y)
    classes = np.unique(y_array)
    
    return x_array, y_array, classes

def get_encoder(classes):
    # encode category from string to numerical values
    classes_encoder = {label:index for index,label in enumerate(classes) }
    
    return classes_encoder

def get_decoder(classes_encoder):
    classes_decoder = {v: k for k, v in classes_encoder.items()}
    
    return classes_decoder


# convert encoded categories for CNN
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
    
def data_preprocessing(data_dir):

    X_train, Y_train, classes = load_data_folder(data_dir)

    classes_encoder = get_encoder(classes)
    classes_decoder = get_decoder(classes_encoder)

    Y_train_encoded = np.array([ classes_encoder[y] for y in Y_train])
    Y_train = convert_to_one_hot(Y_train_encoded, len(classes)).T
    
    return X_train, Y_train, classes_encoder, classes_decoder