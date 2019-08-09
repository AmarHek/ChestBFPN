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
from segmentation_net import *
from model_checkpoint_callback import *
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score


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