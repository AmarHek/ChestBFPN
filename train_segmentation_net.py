# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 2
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import random
import cv2
import time
import numpy as np
import datetime
import glob
import pandas as pd
import warnings
import pickle
import gzip
from PIL import Image
from albumentations import *
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from segmentation_models.backbones import get_feature_layers
from segmentation_models.utils import legacy_support, freeze_model
from segmentation_models.common import ResizeImage
from segmentation_models.utils import extract_outputs, to_tuple
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Add
from keras.layers import ZeroPadding2D
import keras.layers as kl
import keras.backend as K
from keras.callbacks import Callback
from keras.utils.generic_utils import get_custom_objects
from classification_models.weights import weights_collection
from classification_models.resnet.params import get_model_params
from classification_models.utils import load_model_weights
from scipy.stats import entropy


SHAPE_SIZE = 224
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
INPUT_PATH = ROOT_PATH + 'input/DeepChest/'
OUTPUT_PATH = ROOT_PATH + 'modified_data/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
MODELS_PATH = ROOT_PATH + 'models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
HISTORY_FOLDER_PATH = ROOT_PATH + "models/history/"
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)
CACHE_PATH = ROOT_PATH + 'cache/'
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)
PREDICTION_CACHE = OUTPUT_PATH + 'prediction_cache/'
if not os.path.isdir(PREDICTION_CACHE):
    os.mkdir(PREDICTION_CACHE)


global_aug = Compose([
    RandomSizedCrop(min_max_height=(500, 540), width=540, height=540, p=0.8),
    Rotate(limit=5, p=0.5),
    HorizontalFlip(p=0.01),
    OneOf([
        IAAAdditiveGaussianNoise(p=1.0),
        GaussNoise(p=1.0),
    ], p=0.05),
    OneOf([
        MotionBlur(p=0.5),
        MedianBlur(blur_limit=3, p=0.5),
        Blur(blur_limit=3, p=0.5),
    ], p=0.05),
    OneOf([
        IAASharpen(p=1.0),
        IAAEmboss(p=1.0),
    ], p=0.05),
    RandomBrightnessContrast(p=0.01),
    JpegCompression(p=0.01, quality_lower=35, quality_upper=99),
    OneOf([
        ElasticTransform(p=0.5),
        GridDistortion(p=0.5),
    ], p=0.05)
], p=1.0)


def random_augment(image, mask):
    a = global_aug(image=image, mask=mask)
    image = a['image']
    mask = a['mask']

    return image, mask


def batch_generator_train(images_orig, masks_orig, batch_size, preprocess_input, augment=True):
    rng = list(range(len(images_orig)))
    random.shuffle(rng)
    current_point = 0

    while True:
        if current_point + batch_size > len(images_orig):
            random.shuffle(rng)
            current_point = 0

        batch_images = []
        batch_masks = []
        ids = rng[current_point:current_point + batch_size]
        for id in ids:
            img = images_orig[id].copy()
            msk = masks_orig[id].copy()
            if augment:
                img, msk = random_augment(img, msk)

            img = cv2.resize(img, (SHAPE_SIZE, SHAPE_SIZE), interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, (SHAPE_SIZE, SHAPE_SIZE), interpolation=cv2.INTER_LINEAR)
            batch_images.append(np.stack((img, img, img), axis=2))
            batch_masks.append(msk)

        batch_images = np.array(batch_images, dtype=np.float32)
        batch_images = preprocess_input(batch_images)

        batch_masks = np.array(batch_masks, dtype=np.float32)
        batch_masks /= 255.

        current_point += batch_size
        # print(batch_images.shape, batch_masks.shape, batch_images.max(), batch_masks.max())
        yield batch_images, batch_masks


def read_image_files(files, type='train'):
    images = []
    masks = []
    for f in files:
        mask1 = cv2.imread(INPUT_PATH + 'masks_{}/'.format(type) + f)
        mask = np.stack((mask1[:, :, 0], mask1[:, :, 1:].max(axis=2)), axis=2)
        img = cv2.imread(INPUT_PATH + 'Chest X-ray-14/img/'.format(type) + f, 0)
        images.append(img)
        masks.append(mask)
    return images, masks


def preprocess_validation(valid_images, valid_masks, prep_input):
    vi = []
    vm = []
    for i in range(len(valid_images)):
        img = cv2.resize(valid_images[i], (SHAPE_SIZE, SHAPE_SIZE), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(valid_masks[i], (SHAPE_SIZE, SHAPE_SIZE), interpolation=cv2.INTER_LINEAR)
        vi.append(np.stack((img, img, img), axis=2))
        vm.append(msk)

    vi = np.array(vi, dtype=np.float32)
    vi = prep_input(vi)

    vm = np.array(vm, dtype=np.float32)
    vm /= 255.
    print(vi.shape, vm.shape, vi.max(), vm.max())
    return vi, vm


old_args_map = {
    'freeze_encoder': 'encoder_freeze',
    'fpn_layers': 'encoder_features',
    'use_batchnorm': 'pyramid_use_batchnorm',
    'dropout': 'pyramid_dropout',
    'interpolation': 'final_interpolation',
    'upsample_rates': None,  # removed
    'last_upsample': None,  # removed
}


def Conv2DBlock(n_filters, kernel_size,
                activation='relu',
                use_batchnorm=True,
                norm_type='BN',
                name='conv_block',
                **kwargs):
    from keras.layers import Conv2D
    from keras.layers import Activation
    from keras.layers import BatchNormalization
    from keras_contrib.layers.normalization.groupnormalization import GroupNormalization
    from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

    """Extension of Conv2D layer with batchnorm"""
    def layer(input_tensor):

        x = Conv2D(n_filters, kernel_size, use_bias=not(use_batchnorm),
                   name=name+'_conv', **kwargs)(input_tensor)
        if use_batchnorm:
            if norm_type == 'BN':
                x = BatchNormalization(name=name+'_bn',)(x)
            elif norm_type == 'GN':
                x = GroupNormalization(axis=-1, groups=32, name=name+'_gn',)(x)
            elif norm_type == 'IN':
                x = InstanceNormalization(axis=-1, name=name+'_in',)(x)
        x = Activation(activation, name=name+'_'+activation)(x)

        return x
    return layer


def pyramid_block(pyramid_filters=256, segmentation_filters=128, upsample_rate=2,
                  use_batchnorm=False, norm_type='BN', stage=0):
    from keras.layers import Dropout
    """
    Pyramid block according to:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    This block generate `M` and `P` blocks.

    Args:
        pyramid_filters: integer, filters in `M` block of top-down FPN branch
        segmentation_filters: integer, number of filters in segmentation head,
            basically filters in convolution layers between `M` and `P` blocks
        upsample_rate: integer, uspsample rate for `M` block of top-down FPN branch
        use_batchnorm: bool, include batchnorm in convolution blocks

    Returns:
        Pyramid block function (as Keras layers functional API)
    """
    def layer(c, m=None):

        x = Conv2DBlock(pyramid_filters, (1, 1),
                        padding='same',
                        use_batchnorm=use_batchnorm,
                        norm_type=norm_type,
                        name='pyramid_stage_{}'.format(stage))(c)

        if m is not None:
            # print('Up rate: {}'.format(upsample_rate))
            m = Dropout(0.5)(m, training=True)
            up = ResizeImage(to_tuple(upsample_rate))(m)
            x = Add()([x, up])

        # segmentation head
        p = Conv2DBlock(segmentation_filters, (3, 3),
                        padding='same',
                        use_batchnorm=use_batchnorm,
                        norm_type=norm_type,
                        name='segm1_stage_{}'.format(stage))(x)

        p = Conv2DBlock(segmentation_filters, (3, 3),
                        padding='same',
                        use_batchnorm=use_batchnorm,
                        norm_type=norm_type,
                        name='segm2_stage_{}'.format(stage))(p)
        m = x

        return m, p
    return layer


def build_fpn(backbone,
              fpn_layers,
              classes=21,
              activation='softmax',
              upsample_rates=(2,2,2),
              last_upsample=4,
              pyramid_filters=256,
              segmentation_filters=128,
              use_batchnorm=False,
              dropout=None,
              interpolation='bilinear',
              norm_type='BN'):
    """
    Implementation of FPN head for segmentation models according to:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    Args:
        backbone: Keras `Model`, some classification model without top
        layers: list of layer names or indexes, used for pyramid building
        classes: int, number of output feature maps
        activation: activation in last layer, e.g. 'sigmoid' or 'softmax'
        upsample_rates: tuple of integers, scaling rates between pyramid blocks
        pyramid_filters: int, number of filters in `M` blocks of top-down FPN branch
        segmentation_filters: int, number of filters in `P` blocks of FPN
        last_upsample: rate for upsumpling concatenated pyramid predictions to
            match spatial resolution of input data
        last_upsampling_type: 'nn' or 'bilinear'
        dropout: float [0, 1), dropout rate
        use_batchnorm: bool, include batch normalization to FPN between `conv`
            and `relu` layers

    Returns:
        model: Keras `Model`
    """

    from keras.layers import Conv2D
    from keras.layers import Concatenate
    from keras.layers import Activation
    from keras.layers import SpatialDropout2D
    from keras.models import Model

    if len(upsample_rates) != len(fpn_layers):
        raise ValueError('Number of intermediate feature maps and upsample steps should match')

    # extract model layer outputs
    outputs = extract_outputs(backbone, fpn_layers, include_top=True)

    # add upsample rate `1` for first block
    upsample_rates = [1] + list(upsample_rates)

    # top - down path, build pyramid
    m = None
    pyramid = []
    for i, c in enumerate(outputs):
        m, p = pyramid_block(pyramid_filters=pyramid_filters,
                             segmentation_filters=segmentation_filters,
                             upsample_rate=upsample_rates[i],
                             use_batchnorm=use_batchnorm,
                             norm_type=norm_type,
                             stage=i)(c, m)
        pyramid.append(p)


    # upsample and concatenate all pyramid layer
    upsampled_pyramid = []

    for i, p in enumerate(pyramid[::-1]):
        if upsample_rates[i] > 1:
            upsample_rate = to_tuple(np.prod(upsample_rates[:i+1]))
            p = ResizeImage(upsample_rate, interpolation=interpolation)(p)
        upsampled_pyramid.append(p)

    x = Concatenate()(upsampled_pyramid)

    # final convolution
    n_filters = segmentation_filters * len(pyramid)
    x = Conv2DBlock(n_filters, (3, 3), use_batchnorm=use_batchnorm, padding='same')(x)
    if dropout is not None:
        x = SpatialDropout2D(dropout)(x, training=True)

    x = Conv2D(classes, (3, 3), padding='same')(x)

    # upsampling to original spatial resolution
    x = ResizeImage(to_tuple(last_upsample), interpolation=interpolation)(x)

    # activation
    x = Activation(activation)(x)

    model = Model(backbone.input, x)
    return model


def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'he_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    default_bn_params = {
        'axis': 3,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params


def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name


def residual_conv_block(filters, stage, block, strides=(1, 1), attention=None, cut='pre'):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = Add()([x, shortcut])
        return x

    return layer


def residual_bottleneck_block(filters, stage, block, strides=None, attention=None, cut='pre'):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = Conv2D(filters*4, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = Activation('relu', name=relu_name + '2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + '2', **conv_params)(x)

        x = BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = Activation('relu', name=relu_name + '3')(x)
        x = Conv2D(filters*4, (1, 1), name=conv_name + '3', **conv_params)(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = Add()([x, shortcut])

        return x

    return layer


class Slice(kl.Layer):

    def __init__(self, start, stop, **kwargs):
        self.start = start
        self.stop = stop
        super(Slice, self).__init__(**kwargs)

    def call(self, x):
        return x[..., self.start:self.stop]

    def compute_output_shape(self, input_shape):
        bs, h, w, ch = input_shape
        new_ch = self.stop - self.start
        return (bs, h, w, new_ch)

    def get_config(self):
        config = super(Slice, self).get_config()
        config['start'] = self.start
        config['stop'] = self.stop
        return config


def GroupConv2D(filters,
                kernel_size,
                strides=(1, 1),
                groups=32,
                kernel_initializer='he_uniform',
                use_bias=True,
                activation='linear',
                padding='valid',
                **kwargs):
    """
    Grouped Convolution Layer implemented as a Slice,
    Conv2D and Concatenate layers. Split filters to groups, apply Conv2D and concatenate back.

    Args:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer, specifying the stride
            length of the convolution.
        groups: Integer, number of groups to split input filters to.
        kernel_initializer: Regularizer function applied to the kernel weights matrix.
        use_bias: Boolean, whether the layer uses a bias vector.
        activation: Activation function to use (see activations).
            If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
        padding: one of "valid" or "same" (case-insensitive).

    Input shape:
        4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".

    Output shape:
        4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format is "channels_last".
        rows and cols values might have changed due to padding.

    """

    def layer(input_tensor):
        inp_ch = int(K.int_shape(input_tensor)[-1] // groups)  # input grouped channels
        out_ch = int(filters // groups)  # output grouped channels

        blocks = []
        for c in range(groups):
            start = c * inp_ch
            stop = (c + 1) * inp_ch
            x = Slice(start, stop)(input_tensor)
            x = kl.Conv2D(out_ch,
                          kernel_size,
                          strides=strides,
                          kernel_initializer=kernel_initializer,
                          use_bias=use_bias,
                          activation=activation,
                          padding=padding,
                          **kwargs)(x)
            blocks.append(x)

        x = kl.Concatenate(axis=-1)(blocks)
        return x

    return layer


def ChannelSE(reduction=16):
    """
    Squeeze and Excitation block, reimplementation inspired by
        https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py

    Args:
        reduction: channels squeeze factor

    """

    def layer(input_tensor):
        # get number of channels/filters
        channels = K.int_shape(input_tensor)[-1]

        x = input_tensor

        # squeeze and excitation block in PyTorch style with
        # custom global average pooling where keepdims=True
        x = kl.Lambda(lambda a: K.mean(a, axis=[1, 2], keepdims=True))(x)
        x = kl.Conv2D(channels // reduction, (1, 1), kernel_initializer='he_uniform')(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv2D(channels, (1, 1), kernel_initializer='he_uniform')(x)
        x = kl.Activation('sigmoid')(x)

        # apply attention
        x = kl.Multiply()([input_tensor, x])

        return x

    return layer


def SpatialSE():
    """
    Spatial squeeze and excitation block (applied across spatial dimensions)
    """

    def layer(input_tensor):
        x = kl.Conv2D(1, (1, 1), kernel_initializer="he_normal", activation='sigmoid', use_bias=False)(input_tensor)
        x = kl.Multiply()([input_tensor, x])
        return x

    return layer


def ChannelSpatialSE(reduction=2):
    """
    Spatial and Channel Squeeze & Excitation Block (scSE)
        https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66568

    Implementation of Concurrent Spatial and Channel `Squeeze & Excitation` in Fully Convolutional Networks
        https://arxiv.org/abs/1803.02579
    """

    def layer(input_tensor):
        cse = ChannelSE(reduction=reduction)(input_tensor)
        sse = SpatialSE()(input_tensor)
        x = kl.Add()([cse, sse])

        return x

    return layer


get_custom_objects().update({
    'Slice': Slice,
})


def build_resnet(
        repetitions=(2, 2, 2, 2),
        include_top=True,
        input_tensor=None,
        input_shape=None,
        classes=1000,
        block_type='conv',
        attention=None):
    """
    TODO
    """

    import keras.backend as K
    from keras.layers import Input
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    from keras.layers import GlobalAveragePooling2D
    from keras.layers import ZeroPadding2D
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.models import Model
    from keras.engine import get_source_inputs

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # choose residual block type
    if block_type == 'conv':
        residual_block = residual_conv_block
    elif block_type == 'bottleneck':
        residual_block = residual_bottleneck_block
    else:
        raise ValueError('Block type "{}" not in ["conv", "bottleneck"]'.format(block_type))

    # choose attention block type
    if attention == 'sse':
        attention_block = SpatialSE()
    elif attention == 'cse':
        attention_block = ChannelSE(reduction=16)
    elif attention == 'csse':
        attention_block = ChannelSpatialSE(reduction=2)
    elif attention is None:
        attention_block = None
    else:
        raise ValueError('Supported attention blocks are: sse, cse, csse. Got "{}".'.format(attention))

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = 64

    # resnet bottom
    x = BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    x = BatchNormalization(name='bn0', **bn_params)(x)
    x = Activation('relu', name='relu0')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

    # resnet body
    for stage, rep in enumerate(repetitions):
        for block in range(rep):

            filters = init_filters * (2 ** stage)

            # first block of first stage without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = residual_block(filters, stage, block, strides=(1, 1),
                                   cut='post', attention=attention_block)(x)

            elif block == 0:
                x = Dropout(0.5)(x, training=True)
                x = residual_block(filters, stage, block, strides=(2, 2),
                                   cut='post', attention=attention_block)(x)

            else:
                x = residual_block(filters, stage, block, strides=(1, 1),
                                   cut='pre', attention=attention_block)(x)

    x = BatchNormalization(name='bn1', **bn_params)(x)
    x = Activation('relu', name='relu1')(x)

    # resnet top
    if include_top:
        x = GlobalAveragePooling2D(name='pool1')(x)
        x = Dense(classes, name='fc1')(x)
        x = Activation('softmax', name='softmax')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x)

    return model


def _get_resnet(name):
    def classifier(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
        model_params = get_model_params(name)
        model = build_resnet(input_tensor=input_tensor,
                             input_shape=input_shape,
                             classes=classes,
                             include_top=include_top,
                             **model_params)

        model.name = name

        if weights:
            load_model_weights(weights_collection, model, weights, classes, include_top)

        return model
    return classifier


ResNet50 = _get_resnet('resnet50')


class Classifiers:

    _models = {

        'resnet50': [ResNet50, lambda x: x],
    }

    @classmethod
    def names(cls):
        return sorted(cls._models.keys())

    @classmethod
    def get(cls, name):
        """
        Access to classifiers and preprocessing functions

        Args:
            name (str): architecture name

        Returns:
            callable: function to build keras model
            callable: function to preprocess image data

        """
        return cls._models.get(name)

    @classmethod
    def get_classifier(cls, name):
        return cls._models.get(name)[0]

    @classmethod
    def get_preprocessing(cls, name):
        return cls._models.get(name)[1]


def get_backbone(name, *args, **kwargs):
    return Classifiers.get_classifier(name)(*args, **kwargs)


@legacy_support(old_args_map)
def FPN(backbone_name='vgg16',
        input_shape=(None, None, 3),
        input_tensor=None,
        classes=21,
        activation='softmax',
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        pyramid_block_filters=256,
        pyramid_use_batchnorm=True,
        pyramid_dropout=None,
        final_interpolation='bilinear',
        norm_type='BN',
        **kwargs):
    """FPN_ is a fully convolution neural network for image semantic segmentation

    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
                case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
                able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model
                (works only if ``encoder_weights`` is ``None``).
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer (e.g. ``sigmoid``, ``softmax``, ``linear``).
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
                Each of these layers will be used to build features pyramid. If ``default`` is used
                layer names are taken from ``DEFAULT_FEATURE_PYRAMID_LAYERS``.
        pyramid_block_filters: a number of filters in Feature Pyramid Block of FPN_.
        pyramid_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used.
        pyramid_dropout: spatial dropout rate for feature pyramid in range (0, 1).
        final_interpolation: interpolation type for upsampling layers, on of ``nearest``, ``bilinear``.

    Returns:
        ``keras.models.Model``: **FPN**

    .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    """

    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            input_tensor=input_tensor,
                            weights=encoder_weights,
                            include_top=False)

    if encoder_features == 'default':
        encoder_features = get_feature_layers(backbone_name, n=3)

    upsample_rates = [2] * len(encoder_features)
    last_upsample = 2 ** (5 - len(encoder_features))

    model = build_fpn(backbone, encoder_features,
                      classes=classes,
                      pyramid_filters=pyramid_block_filters,
                      segmentation_filters=pyramid_block_filters // 2,
                      upsample_rates=upsample_rates,
                      use_batchnorm=pyramid_use_batchnorm,
                      dropout=pyramid_dropout,
                      last_upsample=last_upsample,
                      interpolation=final_interpolation,
                      activation=activation,
                      norm_type=norm_type)

    if encoder_freeze:
        freeze_model(backbone)

    model.name = 'fpn-{}'.format(backbone.name)

    return model


def get_model(backbone, decoder_type, batch_norm_type, dropout=0.0):
    from segmentation_models import Unet, Linknet, PSPNet

    inp_shape = (SHAPE_SIZE, SHAPE_SIZE, 3)
    classes = 2

    if decoder_type == 'Unet':
        model = Unet(backbone,
                     encoder_weights='imagenet',
                     input_shape=inp_shape,
                     classes=classes,
                     dropout=dropout,
                     norm_type=batch_norm_type,
                     activation='sigmoid')
    elif decoder_type == 'FPN':
        model = FPN(backbone,
                    encoder_weights='imagenet',
                    input_shape=inp_shape,
                    classes=classes,
                    pyramid_dropout=dropout,
                    norm_type=batch_norm_type,
                    activation='sigmoid')
    elif decoder_type == 'Linknet':
        model = Linknet(backbone,
                        encoder_weights='imagenet',
                        input_shape=inp_shape,
                        classes=classes,
                        dropout=dropout,
                        norm_type=batch_norm_type,
                        activation='sigmoid')
    elif decoder_type == 'PSPNet':
        model = PSPNet(backbone,
                       encoder_weights='imagenet',
                       input_shape=inp_shape,
                       classes=classes,
                       psp_dropout=dropout,
                       norm_type=batch_norm_type,
                       activation='sigmoid')
    return model



# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3), protocol=4)


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'), )


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def read_single_image(path):
    try:
        img = np.array(Image.open(path))
    except:
        try:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        except:
            print('Fail')
            return None

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 2:
        img = img[:, :, :1]

    if img.shape[2] == 1:
        img = np.concatenate((img, img, img), axis=2)

    if img.shape[2] > 3:
        img = img[:, :, :3]

    return img


def get_simple_iou_score(mask1, mask2):
    intersection = ((mask1 > 0) & (mask2 > 0)).sum()
    union = ((mask1 > 0) | (mask2 > 0)).sum()
    if union == 0:
        return 1
    return intersection / union


def get_simple_dice_score(mask1, mask2):
    intersection = ((mask1 > 0) & (mask2 > 0)).sum()
    if (mask1.max() > 1) | (mask2.max() > 1):
        print('Dice error!')
        exit()
    sum = mask1.sum() + mask2.sum()
    if sum == 0:
        return 1
    return 2 * intersection / sum


class ModelCheckpoint_IOU(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, filepath_cache, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='max', period=1, patience=None, validation_data=(),
                 pads=(27, 27), thr_list=[0.5]):
        super(ModelCheckpoint_IOU, self).__init__()
        self.interval = period
        self.images_for_valid, self.masks_for_valid, self.preprocess_input = validation_data
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.filepath_cache = filepath_cache
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.monitor_op = np.greater
        self.best = -np.Inf
        self.pads = pads
        self.thr_list = thr_list

        # part for early stopping
        self.epochs_from_best_model = 0
        self.patience = patience

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            score = dict()
            for t in self.thr_list:
                score[t] = 0

            thr = 0.5
            start_time = time.time()
            pred_masks = self.model.predict(self.images_for_valid)
            real_mask = self.masks_for_valid
            pred_mask = pred_masks.copy()
            pred_mask[pred_mask >= thr] = 1
            pred_mask[pred_mask < thr] = 0
            avg_iou = []
            avg_dice = []
            for i in range(pred_mask.shape[0]):
                iou = get_simple_iou_score(pred_mask[i].astype(np.uint8), real_mask[i].astype(np.uint8))
                dice = get_simple_dice_score(pred_mask[i].astype(np.uint8), real_mask[i].astype(np.uint8))
                avg_iou.append(iou)
                avg_dice.append(dice)
            score_iou = np.array(avg_iou).mean()
            score_dice = np.array(avg_dice).mean()

            logs['score_iou'] = score_iou
            logs['score_dice'] = score_dice
            print("IOU score: {:.6f} Dice score: {:.6f} THR: {:.2f} Time: {:.2f}".format(score_iou, score_dice, thr, time.time() - start_time))

            # filepath = self.filepath.format(epoch=epoch + 1, score=score_iou, **logs)
            filepath = self.filepath_cache

            if score_iou > self.best:
                self.epochs_from_best_model = 0
            else:
                self.epochs_from_best_model += 1

            if self.save_best_only:
                current = score_iou
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                        # shutil.copy(filepath, self.filepath_cache)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
                # shutil.copy(filepath, self.filepath_cache)

            if self.patience is not None:
                if self.epochs_from_best_model > self.patience:
                    print('Early stopping: {}'.format(self.epochs_from_best_model))
                    self.model.stop_training = True


def dice_coef(y_true, y_pred):
    from keras import backend as K
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def train_single_model(num_fold, train_files, valid_files, backbone, decoder_type, batch_norm_type):
    from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
    from keras.optimizers import Adam, SGD
    from keras.models import load_model, Model

    restore = 0
    patience = 100
    epochs = 1000
    optim_type = 'Adam'
    learning_rate = 0.0001
    dropout = 0.1
    cnn_type = '{}_{}_{}_{}_drop_{}_baesyan'.format(backbone, decoder_type, batch_norm_type, optim_type, dropout)
    print('Creating and compiling {}...'.format(cnn_type))

    train_images, train_masks = read_image_files(train_files)
    valid_images, valid_masks = read_image_files(valid_files)

    final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
    if os.path.isfile(final_model_path) and restore == 1:
        print('Model already exists for fold {}.'.format(final_model_path))
        return 0.0

    cache_model_path = MODELS_PATH + '{}_temp_fold_{}.h5'.format(cnn_type, num_fold)
    best_model_path = MODELS_PATH + '{}_fold_{}_'.format(cnn_type, num_fold) + '{epoch:02d}-{val_loss:.4f}-iou-{score:.4f}.h5'
    model = get_model(backbone, decoder_type, batch_norm_type, dropout=dropout)
    print(model.summary())
    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)

    loss_to_use = bce_jaccard_loss
    # loss_to_use = jacard_focal_loss
    model.compile(optimizer=optim, loss=loss_to_use, metrics=[iou_score, dice_coef])

    preprocess_input = get_preprocessing(backbone)
    valid_images_1, valid_masks_1 = preprocess_validation(valid_images.copy(), valid_masks.copy(), preprocess_input)

    print('Fitting model...')
    batch_size = 8
    batch_size_valid = 1
    print('Batch size: {}'.format(batch_size))
    steps_per_epoch = len(train_files) // (batch_size)
    validation_steps = len(valid_files) // (batch_size_valid)

    print('Steps train: {}, Steps valid: {}'.format(steps_per_epoch, validation_steps))

    callbacks = [
        # EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint_IOU(best_model_path, cache_model_path, save_best_only=True, verbose=1,
                            validation_data=(valid_images_1, valid_masks_1, preprocess_input), patience=patience),
        # ModelCheckpoint(cache_model_path, monitor='val_loss', verbose=0),
        # ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=5, min_lr=1e-9, min_delta=1e-8, verbose=1, mode='min'),
        CSVLogger(HISTORY_FOLDER_PATH + 'history_fold_{}_{}_lr_{}_optim_{}.csv'.format(num_fold,
                                                                                       cnn_type,
                                                                                       learning_rate,
                                                                                       optim_type), append=True),
    ]

    gen_train = batch_generator_train(train_images, train_masks, batch_size_valid, preprocess_input, augment=True)
    gen_valid = batch_generator_train(valid_images, valid_masks, 1, preprocess_input, augment=False)
    history = model.fit_generator(generator=gen_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=gen_valid,
                                  validation_steps=validation_steps,
                                  verbose=2,
                                  max_queue_size=10,
                                  callbacks=callbacks)

    max_iou = max(history.history['score_iou'])
    best_epoch = np.array(history.history['score_iou']).argmax()

    print('Max IOU: {:.4f} Best epoch: {}'.format(max_iou, best_epoch))

    model.load_weights(cache_model_path)
    model.save(final_model_path)
    now = datetime.datetime.now()
    filename = HISTORY_FOLDER_PATH + 'history_{}_{}_{:.4f}_lr_{}_{}.csv'.format(cnn_type, num_fold, max_iou, learning_rate, now.strftime("%Y-%m-%d-%H-%M"))
    pd.DataFrame(history.history).to_csv(filename, index=False)
    # save_history_figure(history, filename[:-4] + '.png', columns=('jacard_coef', 'val_jacard_coef'))
    return max_iou, cache_model_path


def get_score_on_test_data(model_path, backbone, decoder_type, batch_norm_type, thr=0.5):
    from keras.utils import plot_model
    test_images = []
    test_masks = []
    files = glob.glob(INPUT_PATH + 'masks_test/*.png')
    ITERS_TO_PRED = 1000

    for f in files:
        mask1 = cv2.imread(f)
        mask = np.stack((mask1[:, :, 0], mask1[:, :, 1:].max(axis=2)), axis=2)
        img = cv2.imread(INPUT_PATH + 'Chest X-ray-14/img/' + os.path.basename(f), 0)
        img = cv2.resize(img, (SHAPE_SIZE, SHAPE_SIZE), interpolation=cv2.INTER_LINEAR)
        test_images.append(np.stack((img, img, img), axis=2))
        test_masks.append(mask / 255)

    cache_path = CACHE_PATH + 'preds_cache_v4_all.pkl'
    if not os.path.isfile(cache_path) or 0:
        model = get_model(backbone, decoder_type, batch_norm_type)
        model.load_weights(model_path)
        # plot_model(model, to_file='model.png')
        # exit()

        test_images1 = np.array(test_images, dtype=np.float32)
        preprocess_input = get_preprocessing(backbone)
        test_images1 = preprocess_input(test_images1)
        test_preds_all = []
        for i in range(ITERS_TO_PRED):
            print('Predict: {}'.format(i))
            test_preds = model.predict(test_images1)
            test_preds_all.append(test_preds.copy())
        test_preds_all = np.array(test_preds_all, dtype=np.float32)
        # save_in_file_fast(test_preds, cache_path)
        np.save(cache_path + '.npy', test_preds_all)
        save_in_file_fast((files, test_images, test_masks), cache_path)
    else:
        files, test_images, test_masks = load_from_file_fast(cache_path)
        test_preds_all = np.load(cache_path + '.npy')

    test_preds = test_preds_all.mean(axis=0)
    print(test_preds.shape)

    avg_iou = []
    avg_dice = []

    avg_iou_heart = []
    avg_dice_heart = []

    avg_iou_lungs = []
    avg_dice_lungs = []

    for i in range(test_preds.shape[0]):
        p = test_preds[i]
        print(p.shape)
        p[p > thr] = 255
        p[p <= thr] = 0
        img_mask = cv2.resize(p.astype(np.uint8), (test_masks[i].shape[1], test_masks[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        # img_mask = remove_small_noise_from_mask(img_mask, 10)
        img_mask[img_mask <= 127] = 0
        img_mask[img_mask > 127] = 1

        # show_image(test_masks[i].astype(np.uint8))

        iou = get_simple_iou_score(img_mask.astype(np.uint8), test_masks[i].astype(np.uint8))
        dice = get_simple_dice_score(img_mask.astype(np.uint8), test_masks[i].astype(np.uint8))

        img_mask_exp = np.zeros((img_mask.shape[0], img_mask.shape[1], 3), dtype=np.uint8)
        test_mask_exp = np.zeros((img_mask.shape[0], img_mask.shape[1], 3), dtype=np.uint8)
        img_mask_exp[:, :, :2] = 255 * img_mask.astype(np.uint8)
        test_mask_exp[:, :, :2] = 255 * test_masks[i].astype(np.uint8)

        cv2.imwrite(PREDICTION_CACHE + os.path.basename(files[i]), img_mask_exp)
        cv2.imwrite(PREDICTION_CACHE + os.path.basename(files[i])[:-4] + '_real.png', test_mask_exp)

        # print('Img: {} IOU: {:.4f} Dice: {:.4f}'.format(os.path.basename(files[i]), iou, dice))

        iou_heart = get_simple_iou_score(img_mask[:, :, :1].astype(np.uint8), test_masks[i][:, :, :1].astype(np.uint8))
        dice_heart = get_simple_dice_score(img_mask[:, :, :1].astype(np.uint8), test_masks[i][:, :, :1].astype(np.uint8))

        iou_lungs = get_simple_iou_score(img_mask[:, :, 1:].astype(np.uint8), test_masks[i][:, :, 1:].astype(np.uint8))
        dice_lungs = get_simple_dice_score(img_mask[:, :, 1:].astype(np.uint8), test_masks[i][:, :, 1:].astype(np.uint8))

        avg_iou.append(iou)
        avg_dice.append(dice)

        avg_iou_heart.append(iou_heart)
        avg_dice_heart.append(dice_heart)

        avg_iou_lungs.append(iou_lungs)
        avg_dice_lungs.append(dice_lungs)

    score_iou = np.array(avg_iou).mean()
    score_dice = np.array(avg_dice).mean()

    score_iou_heart = np.array(avg_iou_heart).mean()
    score_dice_heart = np.array(avg_dice_heart).mean()

    score_iou_lungs = np.array(avg_iou_lungs).mean()
    score_dice_lungs = np.array(avg_dice_lungs).mean()

    print("Average IOU score: {:.4f} Average dice score: {:.4f}".format(score_iou, score_dice))
    print("Average IOU heart: {:.4f} Average dice heart: {:.4f}".format(score_iou_heart, score_dice_heart))
    print("Average IOU lungs: {:.4f} Average dice lungs: {:.4f}".format(score_iou_lungs, score_dice_lungs))
    return score_iou_lungs, score_dice_lungs, score_iou_heart, score_dice_heart


def predict_on_other_datasets(model_path, backbone, decoder_type, batch_norm_type, thr=0.5):
    model = get_model(backbone, decoder_type, batch_norm_type)
    model.load_weights(model_path)

    for dataset in ['chexpert', 'china_set', 'jsrt', 'montgomery_set']:
        test_images = []
        files = glob.glob(OUTPUT_PATH + 'dataset_parts/{}/*.png'.format(dataset))
        ITERS_TO_PRED = 1000

        for f in files:
            img = cv2.imread(f, 0)
            img = cv2.resize(img, (SHAPE_SIZE, SHAPE_SIZE), interpolation=cv2.INTER_LINEAR)
            test_images.append(np.stack((img, img, img), axis=2))

        cache_path = CACHE_PATH + 'preds_cache_{}_all_{}.pkl'.format(dataset, ITERS_TO_PRED)
        if not os.path.isfile(cache_path) or 1:
            test_images1 = np.array(test_images, dtype=np.float32)
            preprocess_input = get_preprocessing(backbone)
            test_images1 = preprocess_input(test_images1)
            test_preds_all = []
            for i in range(ITERS_TO_PRED):
                print('Predict: {}'.format(i))
                test_preds = model.predict(test_images1)
                test_preds_all.append(test_preds.copy())
            test_preds_all = np.array(test_preds_all, dtype=np.float32)
            # save_in_file_fast(test_preds, cache_path)
            np.save(cache_path[:-4] + '.npy', test_preds_all)
            save_in_file_fast((files, test_images), cache_path)
        else:
            files, test_images = load_from_file_fast(cache_path)
            test_preds_all = np.load(cache_path + '.npy')


def get_train_val_split():
    random.seed(100)
    cache_path = INPUT_PATH + 'train_val_split.pkl'
    if not os.path.isfile(cache_path):
        files = glob.glob(INPUT_PATH + 'masks_train/*.png')
        print(len(files))
        patients = dict()
        for f in files:
            p = int(os.path.basename(f).split('_')[0])
            if p in patients:
                patients[p].append(os.path.basename(f))
            else:
                patients[p] = [os.path.basename(f)]
        print(len(patients))
        print(patients)
        all_pat = sorted(list(patients.keys()))
        random.shuffle(all_pat)
        test_pat = all_pat[:12]
        train_pat = all_pat[12:]

        train_files = []
        test_files = []
        for t in train_pat:
            train_files += patients[t]
        for t in test_pat:
            test_files += patients[t]
        print(len(train_files), len(test_files))
        save_in_file_fast((train_files, test_files), cache_path)
    else:
        train_files, test_files = load_from_file_fast(cache_path)
    return train_files, test_files


def create_segmentation_model():
    global SHAPE_SIZE

    split = [get_train_val_split()]
    num_split = 0
    res = dict()
    for train_files, valid_files in split:
        num_split += 1
        print('Start Split number {} from {}'.format(num_split, len(split)))
        print('Split files train: ', len(train_files))
        print('Split files valid: ', len(valid_files))

        '''
        backbones = ['vgg16' 'vgg19', 'resnet18', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                     'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'resnext50', 'resnet101',
                     'seresnext50', 'seresnet101', 'senet154', 'densenet121', 'densenet169', 'densenet201',
                     'inceptionv3', 'inceptionresnetv2', 'mobilenet', 'mobilenetv2']
        types = ['Unet', 'FPN', 'Linknet', 'PSPNet']
        norm_types = ['GN', 'IN', 'BN']
        '''

        backbones = ['resnet50']
        types = ['FPN']
        norm_types = ['IN']

        for b in backbones:
            for t in types:
                for nt in norm_types:
                    if t == 'PSPNet':
                        SHAPE_SIZE = 288
                    else:
                        SHAPE_SIZE = 224
                    score, model_path = train_single_model(num_split, train_files, valid_files, b, t, nt)
                    score_iou_lungs, score_dice_lungs, score_iou_heart, score_dice_heart = get_score_on_test_data(model_path, b, t, nt)
                    res[(b, t, nt)] = "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(score, score_iou_lungs, score_iou_heart, score_dice_lungs, score_dice_heart)

    print('Model results: {}'.format(res))


if __name__ == '__main__':
    start_time = time.time()
    create_segmentation_model()
    print('Time: {:.0f} sec'.format(time.time() - start_time))


'''

'''