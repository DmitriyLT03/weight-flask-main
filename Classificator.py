from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.layers import Flatten, Dense, Dropout, Activation, BatchNormalization, Add
from keras.preprocessing import image
import tensorflow as tf
# import os
import cv2
import numpy as np

class Classificator:
    def __init__(self, hidden_dim=128, drop_rate=0.0, sex_threshold=0.05, freeze_backbone=True, weight_path='model_vgg16_seqMT.h5') -> None:
        self.detector = MTCNN()
        self.sex_threshold = sex_threshold
        vgg_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3))
        last_layer = vgg_model.output
        flatten = Flatten()(last_layer)

        if freeze_backbone:
            for layer in vgg_model.layers:
                layer.trainable = False

        def block(flatten, name):
            x = Dense(hidden_dim, name=f'{name}_fc1')(flatten)
            x = BatchNormalization(name=f'{name}_bn1')(x)
            x = Activation('relu', name=f'{name}_act1')(x)
            x = Dropout(drop_rate)(x)
            x = Dense(hidden_dim, name=f'{name}_fc2')(x)
            x = BatchNormalization(name=f'{name}_bn2')(x)
            x = Activation('relu', name=f'{name}_act2')(x)
            x = Dropout(drop_rate)(x)
            return x
        
        x = block(flatten, name='sex')
        out_sex = Dense(1, activation='sigmoid', name='sex')(x)        
        
        # residual connection to age
        x1 = Dense(hidden_dim)(flatten)
        x1 = Add()([x1, x])
        x1 = block(x1, name='age')
        out_age = Dense(1, activation='linear', name='age')(x1)
        
        # residual connection to bmi
        x2 = Dense(hidden_dim)(flatten)
        x2 = Add()([x2, x])
        x2 = block(x2, name='bmi')
        out_bmi = Dense(1, activation='linear', name='bmi')(x2)
        
        custom_vgg_model = tf.keras.Model(vgg_model.input, [out_bmi, out_age, out_sex])
        custom_vgg_model.compile(
            'adam', 
            {'bmi': 'mae','age': 'mae','sex': 'binary_crossentropy'},
            {'sex': 'accuracy'}, 
            loss_weights={'bmi': 0.8, 'age': 0.1, 'sex': 0.1},
        )

        self.model = custom_vgg_model
        self.model.load_weights(weight_path)

    def predict(self, image_path: str) -> dict:
        img_arr = self.open_image(image_path)
        box = self.detect_face(img_arr)
        if not box:
            return {
                "face_status": False
            }
        cropped_img = self.crop(img_arr, *box[0]["box"])
        tmp = self.model.predict(self.resize(cropped_img))
        lst = [float(x[0, 0]) for x in tmp]
        return {
            "face_status": True,
            "data": lst
        }

    def open_image(self, image_path: str) -> np.ndarray:
        """Open image as array"""
        img = image.load_img(image_path)
        img = image.img_to_array(img)
        return img

    def resize(self, arr: np.ndarray) -> np.ndarray:
        """Resize array to (3, 224, 224)"""
        img = cv2.resize(arr, (224, 224))
        img = np.expand_dims(img, 0)
        img = utils.preprocess_input(img)
        return img

    def crop(self, arr: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        return arr[y:(y + h), x:(x + w), :]

    def detect_face(self, arr: np.ndarray) -> dict:
        img = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        box = self.detector.detect_faces(img)
        return box
