from keras import backend as K
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Flatten, BatchNormalization, Input, Convolution1D, MaxPool1D
from keras.optimizers import Adam

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def get_model(weight_path):
    x = Input(shape=(31, 2))
    smile = Convolution1D(4, kernel_size=4, activation='relu', input_shape=(31, 2))(x)
    smile = MaxPool1D()(smile)
    smile = BatchNormalization()(smile)
    smile = Flatten()(smile)
    smile = Dense(64, activation='relu')(smile)
    smile = BatchNormalization()(smile)
    smile = Dense(16, activation='relu')(smile)
    smile = BatchNormalization()(smile)
    smile = Dense(1, activation='sigmoid', name='smile')(smile)

    mount = Convolution1D(4, kernel_size=4, activation='relu', input_shape=(31, 2))(x)
    mount = MaxPool1D()(mount)
    mount = BatchNormalization()(mount)
    mount = Flatten()(mount)
    mount = Dense(64, activation='relu')(mount)
    mount = BatchNormalization()(mount)
    mount = Dense(16, activation='relu')(mount)
    mount = BatchNormalization()(mount)
    mount = Dense(1, activation='sigmoid', name='mouth')(mount)

    model = Model(input=x, output=[smile, mount])

    model.compile(loss={'smile': "binary_crossentropy", 'mouth': "binary_crossentropy"},
                  optimizer=Adam(lr=0.001, decay=0.0001),
                  metrics=[f1],
                  loss_weights=[0.5, 0.5])

    model.load_weights(weight_path)

    return model
