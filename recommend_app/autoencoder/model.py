from django.conf import settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
import cv2
import matplotlib as mpl
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda
from keras.models import Model, load_model
from keras import backend as K
from keras import metrics
from sklearn.model_selection import train_test_split



# data_dir = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "media"), "model_relating_data")
data_dir = os.path.join(settings.MEDIA_ROOT, "model_relating_data")
def crop_center(img, size=156):
    height, width = img.shape[0], img.shape[1]
    rh = int((height - size) / 2)
    rw = int((width - size) / 2)
    return img[rh:rh+size, rw:rw+size, :]

def decode_img(img):
    img = ((img / (img.max() - img.min())) * 255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def cos_similarity(img, data):
    img = np.tile(img, (data.shape[0], 1, 1, 1))
    tmp = np.sum(img * data, axis=(1, 2, 3))
    tmp1 = np.power(np.sum(np.power(img, 2), axis=(1, 2, 3)), 1/2) + 1e-3
    tmp2 = np.power(np.sum(np.power(data, 2), axis=(1, 2, 3)), 1/2) + 1e-3
    cos = tmp / (tmp1 * tmp2)
    return cos

def mse_similarity(img, data):
    mse = np.sum(np.power(data - img, 2), axis=(1, 2, 3))
    return mse

def seek_suggest(enc, train_enc, X_train, train_name, similarity_func):
    similarity = similarity_func(enc, train_enc)
    similarity_arg = np.argsort(similarity)[::-1]
    bs = similarity_arg[0] + np.random.randint(1, 4)
    bs = bs if bs < X_train.shape[0] else X_train.shape[0] - 1
    suggest = X_train[bs]
    name = train_name[bs]
    return suggest, name

def autoencoder(size=(156, 156, 3)):
    inp = Input(shape=size)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    latent_dim = 32
    # 潜在変数
    z_mean = Dense(latent_dim, name='z_mean')(encoded)
    z_log_var = Dense(latent_dim, name='z_log_var')(encoded)

    def sampling(args):
        z_mean, z_log_var = args
        latent_dim = 32
        epsilon_std = 1.0
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0],
                                        K.shape(z_mean)[1],
                                        K.shape(z_mean)[2],
                                        latent_dim),
                                mean=0.,
                                stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    # decoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(z)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    encoder = Model(inputs=[inp], outputs=[encoded])

    # autoencoderの定義
    autoencoder = Model(inputs=[inp], outputs=[decoded])

    # loss関数
    # Compute VAE loss
    xent_loss = K.mean(metrics.binary_crossentropy(inp, decoded), axis=-1)
    kl_loss =  - 0.5 * K.mean(K.sum(1 + K.log(K.square(z_log_var)) - K.square(z_mean) - K.square(z_log_var), axis=-1))
    vae_loss = K.mean(xent_loss + kl_loss)

    autoencoder.add_loss(vae_loss)
    autoencoder.compile(optimizer='adam')

    autoencoder.load_weights(os.path.join(data_dir, "my_first_weights_best.h5"))
    encoder.load_weights(os.path.join(data_dir, "my_first_encoder_weights_best.h5"))

    # https://qiita.com/fukuit/items/2f8bdbd36979fff96b07
    # 0.66
    return autoencoder, encoder

model, encoder = autoencoder()

X_all = np.load(os.path.join(data_dir, "X_all.npy"))
X_name = np.load(os.path.join(data_dir, "X_name.npy"))

X_train, _, train_name, _ = train_test_split(X_all, X_name, random_state=42, test_size=0.05)
X_train = X_train / 255.0

train_data = model.predict(X_train)
train_enc = encoder.predict(X_train)