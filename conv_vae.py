'''
    ------------------------------------
    Author : Bao Jiarong
    Date   : 2020-08-30
    Project: Variational AE (conv)
    Email  : bao.salirong@gmail.com
    ------------------------------------
'''
import tensorflow as tf


class CONV_VAE(tf.keras.Model):
    #................................................................................
    # Constructor
    #................................................................................
    def __init__(self, image_size = 28, latent_dim = 200, filters = 64):
        super(CONV_VAE, self).__init__(name = "CONV_VAE")

        self.image_size= image_size  # height and weight of images
        self.latent_dim= latent_dim

        # Encoder Layers
        self.en_conv0  = tf.keras.layers.Conv2D(filters, kernel_size=(4,4), strides=(2,2), activation = "relu", name = "en_conv0", padding="same")
        self.en_conv1  = tf.keras.layers.Conv2D(filters, kernel_size=(4,4), strides=(2,2), activation = "relu", name = "en_conv1", padding="same")
        self.en_conv2  = tf.keras.layers.Conv2D(filters << 1, kernel_size=(4,4), strides=(2,2), activation = "relu", name = "en_conv2", padding="same")
        # self.en_bn1    = tf.keras.layers.BatchNormalization(name = "en_bn1")
        self.flatten   = tf.keras.layers.Flatten()
        self.en_dense0 = tf.keras.layers.Dense(filters << 3, activation="relu", name = "en_fc0")
        # self.en_bn2    = tf.keras.layers.BatchNormalization(name = "en_bn2")

        # Latent Space
        self.la_dense1= tf.keras.layers.Dense(units = self.latent_dim   , name = "la_fc1")
        self.la_dense2= tf.keras.layers.Dense(units = self.latent_dim   , name = "la_fc2")

        # Decoder Layers
        self.de_dense0 = tf.keras.layers.Dense(filters << 3, activation="relu", name = "de_fc0")
        # self.de_bn1    = tf.keras.layers.BatchNormalization(name = "de_bn1")
        dense_output   = filters * ((self.image_size >> 2) ** 2)
        self.de_dense1 = tf.keras.layers.Dense(dense_output, activation="relu", name = "de_fc1")
        self.de_dense2 = tf.keras.layers.Dense(dense_output, activation="relu", name = "de_fc2")
        # self.de_bn2    = tf.keras.layers.BatchNormalization(name = "de_bn2")
        reshape_output = [self.image_size >> 2, self.image_size >> 2, filters]
        self.reshape   = tf.keras.layers.Reshape(reshape_output, name = "de_main_out")
        self.de_deconv0= tf.keras.layers.Conv2DTranspose(filters << 1, kernel_size=(4,4),strides=(2,2),name="de_deconv0", padding="same")
        self.de_deconv1= tf.keras.layers.Conv2DTranspose(filters << 1, kernel_size=(4,4),strides=(2,2),name="de_deconv1", padding="same")
        self.de_deconv2= tf.keras.layers.Conv2DTranspose(3, kernel_size=(4,4),strides=(2,2),name="de_deconv2", padding="same")
        self.de_act    = tf.keras.layers.Activation("sigmoid")


    #................................................................................
    # Encoder Space
    #................................................................................
    def encoder(self, x, training = None):
        # Encoder Space
        # print(x.shape)
        x = self.en_conv0(x)   #; print(x.shape)
        x = self.en_conv1(x)   #; print(x.shape)
        x = self.en_conv2(x)   #; print(x.shape)
        # x = self.en_bn1(x)   #; print(x.shape)
        x = self.flatten(x)    #; print(x.shape)
        x = self.en_dense0(x)  #; print(x.shape)
        # x = self.en_bn2(x)   #; print(x.shape)
        return x

    #................................................................................
    # Decoder Space
    #................................................................................
    def decoder(self, x, training = None):
        # Decoder Space
        x = self.de_dense1(x)  #; print(x.shape)
        # x = self.de_bn1(x)   #; print(x.shape)
        x = self.de_dense2(x)  #; print(x.shape)
        # x = self.de_bn2(x)   #; print(x.shape)
        x = self.reshape(x)    #; print(x.shape)
        x = self.de_deconv1(x) #; print(x.shape)
        x = self.de_deconv2(x) #; print(x.shape)
        x = self.de_act(x)     #; print(x.shape)
        return x

    #................................................................................
    # Latent Space
    #................................................................................
    def latent_space(self,x):
        mu  = self.la_dense1(x)
        std = self.la_dense2(x)
        shape = mu.shape[1:]
        eps = tf.random.normal(shape, 0.0, 1.0)
        x = mu + eps * (tf.math.exp(std/2.0))
        return x

    #................................................................................
    def call(self, inputs, training = None):
        x = self.encoder(inputs, training)
        x = self.latent_space(x)
        x = self.decoder(x, training)
        return x
