# Task 2: COVID-19 prediction

This task aims to distinguish between participants who reported a COVID-19 positive status and those who reported
testing negative. Note that the positive group may show no symptoms as there are many asymptomatic COVID-19 cases,
while the negative group may show typical respiratory symptoms which are not caused by
an active COVID-19 infection. To reproduce, please follow the steps below:

1. Navigate to the cloned repository (normally, `covid19-sounds-neurips`)
1. Ensure you have downloaded the *task2* files
   - Unzip and copy/move the `data_0426_en_task2.csv` and `0426_EN_used_task2` in *task2* files to
     `./COVID19_prediction/data`
1. OpenSMILE+SVM
   - Go to the path `cd ./COVID19_prediction/Opensmile`
   - Extract features `python 1_extract_opensmile.py`.
    Note: as above, you can skip this by copying the extracted feature csv from `task2/opensmile` files to this path.
   - Classification `python 2_classifcation.py`
1. Pre-trained VGGish
   - Prepare input

     ```shell
      cd ./COVID19_prediction/data
      python pickle_data.py
     ```

   - Go to model's path `cd ./COVID19_prediction/COVID_model`
   - Train the model `sh run_train_frozen.sh`
   - Test the model `sh run_test_frozen.sh`
1. Fine-tuned VGGish
   - Prepare input

     ```shell
      cd ./COVID19_prediction/data
      python pickle_data.py
     ```

     Note: If you have done already this skip in the pre-trained VGGish model, you can skip it here.
   - Go to model's path `cd ./COVID19_prediction/COVID_model`
   - Train the model `sh run_train.sh`
   - Test the model `sh run_test.sh`

Note, you should have two files in the data folder, namely:

- Raw audio: `./COVID19_prediction/data/0426_EN_used_task2`
- Meta data and split: `./COVID19_prediction/data/data_0426_en_task2.csv`

Which, as described, you can pre-process using the following command:

```bash
python pickle_data.py
```

The results of this task are summarised in Table 3:

![model](../assets/table3.png)
