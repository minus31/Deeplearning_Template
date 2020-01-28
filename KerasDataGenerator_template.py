import numpy as np
import tensorflow as tf
import cv2
from augmentation import Augment
from preprocess import input_preprocess, label_preprocess

# Initialize portrait augment
aug = Augment()
## Define augment params
aug_params = {
    "angle_range": (-45, 45),
    "scale_range": (0.6, 1.5),
    "gamma_range": (0.5, 1.5)
}

def get_stopfile(filename='stopfiles.txt')
    with open(filename, 'r') as f:

        stopfiles = f.read()
        stopfiles = stopfiles.split("\n")

    return stopfiles 


class DataGeneratorSample(tf.keras.utils.Sequence):
    'Generate data for Keras'

    def __init__(self, 
                 file_list, 
                 batch_size=32, 
                 dim=(512, 512), 
                 n_channels=3, 
                 shuffle=True, 
                 augment=False, 
                 stopfile=None,
                 train=True):

        self.dim = dim
        self.batch_size = batch_size
     
        if stopfile:
            sf = get_stopfile(filename=stopfile)
            file_list = [f for f in file_list if f not in sf]
            
        self.file_list = np.array(file_list)
        self.total_numOfFiles = self.file_list.shape[0]
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()
        self.train = train
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_list))
        
        if self.shuffle == True:
#             np.random.seed(7777)
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.total_numOfFiles // self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]
        file_list_temp = self.file_list[indexes]
        
        # Generate data
        X, y = self.__data_generation(file_list_temp)
        return X, y

    def __get_data(self, img_path, label_path):
        """
        read_single dataset 
        """
        
        # Load img & label
        h, w = self.dim

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        if self.augment:

            img, label = aug.augment(img, label, aug_params)
        
        # Manage preprocessing logic in a different file
        img = input_preprocess(img, (w, h))

        label = output_preprocess(label, (w, h))
        
        return img, label

    def __data_generation(self, file_list_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, self.dim[0], self.dim[1], 1))

        # Generate data
        for idx, f in enumerate(file_list_temp):
            
            img_path = f
            label_path = f.replace("/img/", "/label/")
            
            X[idx], y[idx] = self.__get_data(img_path=img_path, 
                                               label_path=label_path)             
        return X, y
