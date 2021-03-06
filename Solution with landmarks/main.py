import gdown
import os
import glob
import pandas as pd
from keras import backend as K
import tensorflow as tf
from keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
import gc
import sys
import numpy as np
import cv2 as cv
from model import get_model
from tools import get_face, get_my_points
import imutils

def load_weights():
    print('load weights ...')
    gdown.download('https://drive.google.com/uc?id=16RH0SBffeXsetExQZAUWEYB0ZzjUpNAa', 'model.h5', quiet=True)
    return 'model.h5'


def predict(model, test_filelist):
    print('predict ...')
    predict_df = pd.DataFrame()

    test_imgs = [get_face(imread(i))[0] for i in test_filelist]
    test_imgs = [imutils.resize(i, width=256) for i in test_imgs]
    test_imgs = [get_my_points(i) for i in test_imgs]

    pred = model.predict(np.array(test_imgs))
    predict_df['filename'] = list(map(lambda x: x.split("/")[-1], test_filelist))
    predict_df['smile'] = pred[0]
    predict_df['open_mouth'] = pred[1]

    predict_df.to_csv(os.path.join(path_submission, 'predict.csv'), index=False)

    print('File predict.csv created in ' + path_submission)


path_models = 'models'
path_test = sys.argv[1]
path_submission = sys.argv[1]
img_rows = 224
img_cols = 224

if __name__ == '__main__':

    model = get_model(load_weights())

    test_filelist = glob.glob(os.path.join(path_test, '*.jpg'))

    predict(model, test_filelist)

