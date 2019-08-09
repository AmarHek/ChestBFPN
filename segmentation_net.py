# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import numpy as np
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
