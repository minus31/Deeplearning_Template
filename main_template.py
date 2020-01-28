import os
import time
import argparse

import numpy as np
import cv2

import tensorflow as tf

from data_generator import DataGenerator
from metrics import *
from models import *
import os
import time
import argparse


import warnings
warnings.filterwarnings("ignore")

def get_current_day():
        import datetime
        now = datetime.datetime.now()
        return now.strftime('%Y%m%d')
    
def load_fileList(dataset_path):
    """NOT IMPLEMENTED"""
    return 
    
# Seed value setting 
_seed = 7777
np.random.seed(_seed)
tf.random.set_seed(_seed)

class DeepLearningFrameWork():
    def __init__(self, config):
    
        self.input_shape = (config.input_shape, config.input_shape, 3)
        self.dataset_path = config.dataset_path
        
        if config.train:
            self.batch_size = config.batch_size
            self.nb_epoch = config.nb_epoch
            self.lr = config.lr

        self.finetune = config.finetune            
        self.val_ratio = config.val_ratio# default=0.8
        self.model = config.model
        self.checkpoint = config.checkpoint # default=100
        
        # default="trained_models/{get_current_day()}"
        self.checkpoint_path = os.path.join(config.checkpoint_path, get_current_day()) 
    
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
            
        # default=None
        self.weight_dir = config.weight_dir

        self.img_paths = load_fileList(self.dataset_path)
        
        
    def build_model(self):

        self.model = network(input_size=self.input_shape, train=self.train)

        
    def train(self):

        if self.finetune:
            print('load pre-trained model weights')
            self.model.load_weights(self.weight_dir, by_name=True)

        train_params = {
            'dim': self.input_shape[:2],
            'batch_size': self.batch_size,
            'n_channels': self.input_shape[-1],
            'shuffle': True,
            'augment': True,
            'train':True,
        }

        test_params = {
            'dim': self.input_shape[:2],
            'batch_size': self.batch_size,
            'n_channels': self.input_shape[-1],
            'shuffle': True,
            'augment': False,
            'train':False,
        }
        
        img_paths = self.img_paths
        
        self.train_img_paths = np.random.choice(img_paths, int(img_paths.shape[0] * self.val_ratio), replace=False)
        self.test_img_paths = np.setdiff1d(img_paths, self.train_img_paths)

        train_gen = DataGenerator(self.train_img_paths, **train_params)
        test_gen = DataGenerator(self.test_img_paths, **test_params)

        opt = tf.keras.optimizers.adam(lr=self.lr)

        self.model.compile(
                      loss={"output" : ce_dice_focal_combined_loss,
                            "boundary_attention" : "binary_crossentropy",},
                      loss_weights=[0.8, 0.2],
                      optimizer=opt,
                      metrics={"output" : [iou_coef, "mse"]})

        """ Callback """
        monitor = 'loss'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
        """Callback for Tensorboard"""
        tb = keras.callbacks.TensorBoard(log_dir="./logs/", update_freq='batch')
        """Callback for save Checkpoints"""
        mc = keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoint_path, '{epoch:02d}-{val_loss:.2f}.h5'), 
                                                verbose=1, 
                                                monitor='val_loss',
                                                save_weights_only=True)

        """ Training loop """
        STEP_SIZE_TRAIN = len(self.train_img_paths) // train_gen.batch_size
        STEP_SIZE_VAL = len(self.test_img_paths) // test_gen.batch_size
        t0 = time.time()

        for epoch in range(self.nb_epoch):
            t1 = time.time()
            res = self.model.fit_generator(generator=train_gen,
                                      validation_data=test_gen,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      validation_steps = STEP_SIZE_VAL,
                                      initial_epoch=epoch,
                                      epochs=epoch + 1,
                                      callbacks=[reduce_lr, tb, mc],
                                      verbose=1,
                                      shuffle=True)
            t2 = time.time()
            # print(res.history)

            print('Training time for one epoch : %.1f' % ((t2 - t1)))

            # checkpoint마다 id list를 섞어서 train, Val generator를 새로 생성
            if (epoch + 1) % self.checkpoint == 0:
                print("shuffle the datasets")
                self.train_img_paths = np.random.choice(img_paths, int(img_paths.shape[0] * self.val_ratio), replace=False)
                self.test_img_paths = np.setdiff1d(img_paths, self.train_img_paths)

                train_gen = DataGeneratorMatting(self.train_img_paths, **train_params)
                test_gen = DataGeneratorMatting(self.test_img_paths, **test_params)
                
        print("Entire training time has been taken {} ", t2 - t0)

        return None


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--input_shape', type=int, default=256)
    args.add_argument('--nb_epoch', type=int, default=1000)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--lr', type=float, default=0.00045)
    args.add_argument('--val_ratio', type=float, default=0.8)

    args.add_argument('--model', type=str, default="mattingnet")
    args.add_argument('--checkpoint', type=int, default=100)
    args.add_argument('--checkpoint_path', type=str, default="./trained_models")
    args.add_argument('--weight_dir', type=str, default="")
    args.add_argument('--img_path', type=str, default="")
    args.add_argument('--tflite_name', type=str, default="")

    args.add_argument('--train', type=bool, default=False)
    args.add_argument('--finetune', type=bool, default=False)
    args.add_argument('--infer_single_img', type=bool, default=False)
    args.add_argument('--convert', type=bool, default=False)
    args.add_argument('--android', type=bool, default=False)

    config = args.parse_args()

    DFW = DeepLearningFrameWork(config)

    if config.train:
        DFW.train()

    if config.infer_single_img:
        print("NOT IMPLEMETED")
        # pred = DFW.infer_single_img(config.img_path)
        # print(pred)