from keras import backend as K
from keras.applications import NASNetMobile
from keras.layers import Dense, Dropout, BatchNormalization, GlobalMaxPooling2D
from keras.regularizers import l2
import tensorflow as tf

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

    model = NASNetMobile(weights = "imagenet", include_top=False, input_shape = (224, 224, 3))
    #Adding custom Layers
    x = model.output
    x = BatchNormalization()(x)
    x = GlobalMaxPooling2D()(x)

    x1 = Dense(1024, activation="elu", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(256, activation="elu", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Dropout(0.5)(x1)

    x2 = Dense(1024, activation="elu", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(256, activation="elu", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(x2)
    x2 = BatchNormalization()(x2)
    #x2 = Dropout(0.5)(x2)

    out_smile = Dense(1, activation="sigmoid", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), name='out_smile')(x1)
    out_mouth = Dense(1, activation="sigmoid", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), name='out_mouth')(x2)

    from keras.models import Model
    # creating the final model
    model_final = Model(input = model.input, output = [out_smile, out_mouth])

    from keras.optimizers import Adam

    # compile the model
    model_final.compile(loss={'out_smile': "binary_crossentropy", 'out_mouth': "binary_crossentropy"},
                        optimizer=Adam(lr=0.001, decay=0.0001), metrics=[f1], loss_weights=[0.5, 0.5])

    model_final.load_weights(weight_path)

    return  model_final