<p>Solution to Kaggle's Airbus Ship Detection Challenge for the test task.</p>

<p>The goal is to analyze satellite images of container ships and produce segmentation masks of the ships.</p>

<p>Initial dataset of the challenge consists of "train_v2" and "train_v2" folders with images for training and testing, 
the train_ship_segmentations_v2.csv file (provides the ground truth for the training on images in the "train_v2"),
the sample_submission_v2.csv file (contains list of test images in "test_v2"). "train_v2" and "train_v2" are ignored in git.</p>

<p>To manage data from .csv files 'pandas' library is used</p>

<p>Unet architecture is used for neural network.</p>

<p>The Unet model is built using tf.keras.</p>

<p>The solution is split into three main part:</p>

<ul style="list-style: none;">
    <li>1. Unet model construction and saving (model.py)</li>
    <li>2. Model training (model_training.py)</li>
    <li>3. Model inference (model_inference.py)</li>
</ul>

<p>To build the model execute "model.py". It saves the model in "model_arck.h5" file.</p>

<p>To train the model execute "model_training.py". It loads previously built model from "model_arck.h5" 
and performs training on it using tf.keras.Model.fit(...). The process of creating image and 
the ground-truth arrays for this method showed in "exploratory_data_analysis_of_the_dataset.ipynb". 
Just in case the process interrupts, model saves checkpoints.
Then trained model is saved in "model.h5" file.</p>

<p>This process take a lot of time and requires powerful computational resources. That's why in this solution, 
in order to demonstrate model performance, it is trained on a small set of images, because computational
capacities of the device I'm using doesn't allow me to train the model on the entire dataset
which is made up of 192 556 images. In this project the model is trained on a set of 100 images.</p>

<p>"model_inference.py" contains functions for model testing such as:</p>

<ul style="list-style: none;">
    <li>Test_all_in_folder - for plotting prediction of every image in a given folder one by one</li>
    <li>Test_with_dice_score - for evaluating dice score between prediction and ground-truth mask. 
    Input images must origin from "train_v2" in order to be able to extract masks for them. Otherwise the function won't work.</li>
    <li>fill_submission_file - for filling submission file with model predictions results on images in "test_v2".</li>
</ul>

<p>Filling the submission file is one of the main requirements of this challenge. Unfortunately the performance of fill_submission_file function turned out to be slow due to the large number of the test images (15 606) and possible incorrect behavior of the model as it's poorly trained. For demonstration, this function is applied for the top 20 values in 'sample_submission_v2.csv' as shown in "exploratory_data_analysis_of_the_dataset.ipynb".</p>