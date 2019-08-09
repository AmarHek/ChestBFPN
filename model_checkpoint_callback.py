# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import warnings
import numpy as np
import time
from keras.callbacks import Callback


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

