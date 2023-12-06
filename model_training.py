import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm 
from skimage.io import imread
import pandas as pd

#Image size
IMG_WIDTH = 768
IMG_HEIGHT = 768
IMG_CHANNELS = 3

#Loading previously saved model
model = tf.keras.models.load_model('model.h5')

#Path to the set of images for training model
#An input image must be square sized
TRAIN_PATH = 'train/'

#Extracting names of the images which will be used for identification
train_ids = next(os.walk(TRAIN_PATH))[2]

#X_train will contain extracted images
#Y_train will contain masks (the ground-truth dataset)
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)

#Extracts mask pixels from the dataset
def get_mask_coords(img_id, mask_dataframe):
    def group_mask_coords(mask_crds):
        temp_mask=[]
        for i in range(len(mask_crds)):
            if i%2 == 0:
                temp_mask.append([mask_crds[i],mask_crds[i+1]])
        return temp_mask

    subdf = mask_dataframe.loc[(mask_dataframe['ImageId'] == img_id)]
    mask_coords=""
    for row in range(len(subdf)):
        tmp = subdf.iloc[row]['EncodedPixels']
        mask_coords += (tmp if pd.notna(tmp) else "") + ' '
    mask_coords = list(map(int, mask_coords.split()))
    mask_coords = group_mask_coords(mask_coords)
    return mask_coords

#Fills mask array with boolean values based on coordinates of encoded pixels in the dataset
def fill_mask(mask_crds):
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
    for m_crd in mask_crds:
        for i in range(m_crd[1]):
            x, y = np.divmod((m_crd[0] + i), IMG_HEIGHT)
            mask[y][x] = True
    return mask

#Filling X_train and Y_train with images and masks
print('Loading training images and masks')
mask_data_frame = pd.read_csv('train_ship_segmentations_v2.csv')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    X_train[n] = img  #Fill empty X_train with values from img
    Y_train[n] = fill_mask(get_mask_coords(id_, mask_data_frame))


#Use callbacks to log training events and checkpoints just in case the training process interrupts
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_checkpoint.h5', verbose=1, save_best_only=True, monitor='val_loss')
callbacks = [checkpointer, tf.keras.callbacks.TensorBoard(log_dir='logs')]

#Start training
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=5, epochs=20, callbacks=callbacks)

#Save trained model
model.save('model.h5')