# PAN 2021 Authorship Verification Model

This repository contains the source code used to generate my model for the PAN 2021 authorship verification shared task: https://pan.webis.de/clef21/pan21-web/author-identification.html

I have included the trained models that were used in my submission. However, the training datasets are not included in the repository and can be downloaded from: https://pan.webis.de/data.html


## Note
I developed most scripts in Jupyter Notebooks. Once I'm comfortable with the code, some long running processes (such as preprocessing datasets) were run on an NYU High Performance Computing cluster. Files with names with `_hpc` are meant to be run in such clusters where the processing is divided across multiple machines.

## Reproducing Results
1. Download the datasets and place them in `data` directory. Small dataset should be places in `data/small` and large dataset in `data/large`.
2. Split the dataset into train and test sets. This is done in `small_dataset_splitting.ipynb` and 'large_dataset_splitting.ipynb'. This creates `temp_data/small_model_training_data/dataset_partition.p` and `temp_data/large_model_training_data/dataset_partition.p`. These files are included in the repository.
3. Preprocessing: To preprocess the data, run `small_preprocess.ipynb` and `large_preprocess.ipynb`. This process takes a long time and I did this on the HPC cluster. The code for that is available at `preprocess_hpc.py` and the code merge the preprocessed data from the HPC is available in the jupyter notebooks. Preprocessed data are not included in this repo
4. Model training: Run `small_train_model.ipynb` and `large_train_model.ipynb`. These scripts read the preprocessed files, create the feature vectors (and saves them as numpy memory mapped arrays), and trains the model. The feature vectorizers and the finals models are included in the repository. We used the PAN 2021 evaluator script to compute all the performance metrics (Script obtained from: https://github.com/pan-webis-de/pan-code/blob/master/clef21/authorship-verification/pan20_verif_evaluator.py). The same code, developed to run on the HPC is available on `train_model_hpc.py`.
5. Predicting: `large_predict.py` and `small_predict,py` are the files used to make predictions using the trained model on TIRA.

Please contact janith@nyu.edu if you have trouble running these scripts.  