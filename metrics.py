from tensorflow.keras import backend as K
import tensorflow as tf

def binary_focal_loss(gamma=3., alpha=.7):
    def binary_focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def mean_iou(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

def dice(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
    num = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    den = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) 
    return K.mean((2*num + K.epsilon()) / (den + K.epsilon()))

def mean_iou_neg(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    
    y_pred_neg = tf.ones(tf.shape(y_pred), dtype='float32') - y_pred
    y_true_neg = tf.ones(tf.shape(y_pred), dtype='float32') - y_true
    inter_neg = K.sum(K.sum(K.squeeze(y_true_neg * y_pred_neg, axis=3), axis=2), axis=1)
    union_neg = K.sum(K.sum(K.squeeze(y_true_neg + y_pred_neg, axis=3), axis=2), axis=1) - inter_neg
    
    iou_class = inter/(union+ K.epsilon())
    iou_background = inter_neg/(union_neg+ K.epsilon())
    
    return K.mean((iou_class + iou_background)/2)

def dice_neg(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
    num = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    den = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) 
    
    y_pred_neg = tf.ones(tf.shape(y_pred), dtype='float32') - y_pred
    y_true_neg = tf.ones(tf.shape(y_pred), dtype='float32') - y_true
    num_neg = K.sum(K.sum(K.squeeze(y_true_neg * y_pred_neg, axis=3), axis=2), axis=1)
    den_neg = K.sum(K.sum(K.squeeze(y_true_neg + y_pred_neg, axis=3), axis=2), axis=1)
    
    dice_class = 2 * num / (den + K.epsilon())
    dice_background = 2 * num_neg / (den_neg + K.epsilon())
    
    return K.mean((dice_class + dice_background)/2)

def mean_iou_weighted(y_true, y_pred, alpha = 0.6):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    
    y_pred_neg = tf.ones(tf.shape(y_pred), dtype='float32') - y_pred
    y_true_neg = tf.ones(tf.shape(y_pred), dtype='float32') - y_true
    inter_neg = K.sum(K.sum(K.squeeze(y_true_neg * y_pred_neg, axis=3), axis=2), axis=1)
    union_neg = K.sum(K.sum(K.squeeze(y_true_neg + y_pred_neg, axis=3), axis=2), axis=1) - inter_neg
    
    iou_class = inter/(union+ K.epsilon())
    iou_background = inter_neg/(union_neg+ K.epsilon())
    
    return K.mean(alpha * iou_class + (1-alpha) * iou_background)


def dice_weighted(y_true, y_pred, alpha = 0.6):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
    num = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    den = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) 
    
    y_pred_neg = tf.ones(tf.shape(y_pred), dtype='float32') - y_pred
    y_true_neg = tf.ones(tf.shape(y_pred), dtype='float32') - y_true
    num_neg = K.sum(K.sum(K.squeeze(y_true_neg * y_pred_neg, axis=3), axis=2), axis=1)
    den_neg = K.sum(K.sum(K.squeeze(y_true_neg + y_pred_neg, axis=3), axis=2), axis=1)
    
    dice_class = 2 * num / (den + K.epsilon())
    dice_background = 2 * num_neg / (den_neg + K.epsilon())
    
    return K.mean(alpha * dice_class + (1-alpha) * dice_background)