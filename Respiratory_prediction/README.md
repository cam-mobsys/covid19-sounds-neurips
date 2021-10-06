# Task 1: Respiratory symptom prediction

This task aims at exploring the potential of various sound types in predicting respiratory abnormalities, where the
symptomatic group consists of participants who reported any respiratory symptoms, including dry cough, wet
cough, fever, sore throat, shortness of breath, runny nose, headache, dizziness, and chest
tightness, while asymptomatic controls are those who reported no symptoms. To reproduce, please follow the steps below:

1. Navigate to the cloned repository (normally, `covid19-sounds-neurips`)
1. Ensure you have downloaded the *task1* files from Google Drive
   - Unzip and copy/move the `data_0426_en_task1.csv` and `0426_EN_used_task1` in *task1* files to
     `./Respiratory_prediction/data`
1. OpenSMILE+SVM
   - Go to the path `cd ./Respiratory_prediction/Opensmile`
   - Extract features `python 1_extract_opensmile.py` Note, you can skip this, see below.
   - Perform classification `python 2_classifcation.py`
1. Pre-trained VGGish
   - Prepare input:

     ```shell
      cd ./Respiratory_prediction/data
      python pickle_data.py
      python pickle_user.py
     ```

   - Go to model's path `cd ./Respiratory_prediction/model`
   - Train the model `sh run_train_frozen.sh`
   - Test the model `sh run_test_frozen.sh`
1. Fine-tuned VGGish
   - Prepare input:

     ```shell
      cd ./Respiratory_prediction/data
      python pickle_data.py
      python pickle_user.py
     ```

     Note: If you have done already this skip in the pre-trained VGGish model, you can skip it here.
   - Go to model's path `cd ./Respiratory_prediction/model`
   - Train the model `sh run_train.sh`
   - Test the model `sh run_test.sh`

Running the `1_extract_opensmile.py` script requires [openSMILE][7]. However, you can skip this by copying the already
extracted feature `csv` from `task1/opensmile` files to this path (namely, `./Respiratory_prediction/Opensmile`).

Note, you should have two files in the data folder, namely:

- Raw audio: `./COVID19_prediction/data/0426_EN_used_task1`
- Meta data and split: `./COVID19_prediction/data/data_0426_en_task1.csv`

Which, as described, you can pre-process using the following command:

```bash
python pickle_data.py
```

Results are summarised in Table 2:

![model](../assets/table2.png)
