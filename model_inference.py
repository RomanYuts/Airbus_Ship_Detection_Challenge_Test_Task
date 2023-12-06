import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import pandas as pd

IMG_WIDTH = 768
IMG_HEIGHT = 768
IMG_CHANNELS = 3

model = tf.keras.models.load_model('model.h5')

TEST_PATH = 'test/'

#Function for testing the test image set provided in the .zip file of the challange (test_v2)
#and filling the submission form with the results
#Performing of this function might be slow due to the big number of the test images (15 606) 
#and possible incorrect behavior of the model
#the 'top' parameter takes top n images in the submission form for testing
def fill_submission_file(target_file, source_file, test_path, model, top):
    #Transforms model prediction data into mask points for submission file format
    def predicted_to_coords(pred_):
        crds_str = ""
        i = j = 0
        while(i < np.shape(pred_)[0]):
            j = 0
            while(j < (np.shape(pred_)[1] - 1)):
                if pred_[i][j] == True:
                    if crds_str != "":
                        start_point = ' '
                    else:
                        start_point = ""
                    start_point += str((i + 1)*np.shape(pred_)[0] + (j + 1))
                    k = 0
                    while((pred_[i][j] == True) & (j < (np.shape(pred_)[1] - 1))):
                        k +=1
                        j +=1
                    crds_str += start_point + ' ' + str(k)
                else:
                    j += 1
            i += 1

        return crds_str
    
    submission_data_frame = pd.read_csv(source_file)
    for row in tqdm(range(top)):   
        img_path = test_path + submission_data_frame.iloc[row]['ImageId']
        img = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        img[0] = imread(img_path)[:,:,:IMG_CHANNELS]
        pred = model.predict(img, verbose=0)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = pred.astype(np.bool_)
        submission_data_frame.iloc[row]['EncodedPixels'] = predicted_to_coords(pred[0])
    submission_data_frame.to_csv(target_file, index=False)

#Function for testing all images in the TEST_PATH folder 
#and plots prediction for every image in a given folder one by one
def Test_all_in_folder(path, model):
    test_ids = next(os.walk(path))[2]
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    print('Testing') 

    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        img_path = path + id_
        img = imread(img_path)[:,:,:IMG_CHANNELS]
        X_test[n] = img

    preds_test = model.predict(X_test, verbose=1)

    for i in range(len(X_test)):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(np.squeeze(preds_test[i]))
        ax2.imshow(np.squeeze(X_test[i]))
        ax1.title.set_text('Prediction')
        ax2.title.set_text(test_ids[i])
        plt.tight_layout()
        plt.show()

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

#Calculating the dice score
def dice_score(im1, im2):

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

#Testing images taken from the train dataset (train_v2 in the .zip file of the challenge) for calculating dice score
def Test_with_dice_score(dataset_file, img_path, model, plot):
    test_ids = next(os.walk(img_path))[2]
    data_set = pd.read_csv(dataset_file)
    data_set = data_set.loc[(data_set['EncodedPixels'].notnull())]
    mask = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
    images = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    for i in range(len(test_ids)):
        mask[i] = fill_mask(get_mask_coords(test_ids[i], data_set))
        images[i] = imread(img_path + test_ids[i])[:,:,:IMG_CHANNELS]
    pred = model.predict(images, verbose=0)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = pred.astype(np.bool_)
    for i in range(len(test_ids)):
        dice = dice_score(mask[i], pred[i])
        print(" Image name: " + test_ids[i] + "; Dice Score: " + str(dice))
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(pred[i])
            ax2.imshow(mask[i])
            ax1.title.set_text('Prediction')
            ax2.title.set_text('Mask')
            plt.tight_layout()
            plt.gcf().text(0.05, 0.9, "Dice Score: " + str(dice), fontsize=12)
            plt.show()



#fill_submission_file('submission.csv', 'sample_submission_v2.csv', 'test_v2/', model, 20)

Test_all_in_folder(TEST_PATH, model=model)

#Test_with_dice_score('train_ship_segmentations_v2.csv', 'test/', model=model, plot = True)