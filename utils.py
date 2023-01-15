# Utilities for SimCLRv2
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers

def get_model(name="resnet50"):
    return tf.keras.applications.resnet.ResNet50(weights=None, 
    include_top=False, input_shape=(224, 224, 3))

def augment_image(image, s=1):
    # Random crop
    x = tf.image.random_crop(
        image, size=[224 - 0.8 * s, 224 - 0.8 * s, 3])

    # Color jitter (Credits: https://github.com/sayakpaul/SimCLR-in-TensorFlow-2/blob/master/SimCLR_ImageNet_Subset.ipynb)
    x = tf.image.random_brightness(x, max_delta=0.8*s)
    x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
    x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
    x = tf.image.random_hue(x, max_delta=0.2*s)
    x = tf.clip_by_value(x, 0, 1)
    
    # Gaussian blur
    x = tfa.image.gaussian_filter2d(x, sigma=0.8*s)
    return x

def NTXent(y1, y2, t=1.0, batch_size=32):
    '''
    Steps:
    1. Find numerator (simple cosine similiarity)
    2. Concat, mask and find denominator
    '''
    numerator = tf.keras.losses.CosineSimiliarity(axis=1)(y1, y2) / t

    negative_mask = tf.ones((batch_size, 2 * batch_size), dtype=tf.uint8)

    for i in batch_size:
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0


    concat_sim_mat = tf.concat([y1, y2], axis=0)

    
    

