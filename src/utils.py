import re as re
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob


def load_dataset_data():
    train_data = "../data/HAPT Data Set/Train/X_train.txt"
    train_labels = "../data/HAPT Data Set/Train/y_train.txt"
    train_subjects = "../data/HAPT Data Set/Train/subject_id_train.txt"

    test_data = "../data/HAPT Data Set/Test/X_test.txt"
    test_labels = "../data/HAPT Data Set/Test/y_test.txt"
    test_subjects = "../data/HAPT Data Set/Test/subject_id_test.txt"

    feature_names = "../data/HAPT Data Set/features.txt"
    
    X_train, y_train, subject_train = read_dataset_data(train_data, train_labels, train_subjects, feature_names)
    X_test, y_test, subject_test = read_dataset_data(test_data, test_labels, test_subjects, feature_names)
    return X_train, y_train, subject_train, X_test, y_test, subject_test

def read_dataset_data(data, labels, subjects, feature_names):
    feature_names = open(feature_names).read().splitlines()
    feature_names = [feature.replace(' ', '')+f"_{idx}" for idx, feature in enumerate(feature_names)]

    X = pd.read_csv(data, sep=" ", header=None, names=feature_names)
    y = pd.read_csv(labels, sep=" ", header=None, names=['activity_label'])
    subjects = pd.read_csv(subjects, sep=" ", header=None, names=["subjects"])

    return X, y, subjects

def load_raw_data(raw_data = "../data/HAPT Data Set/RawData/*.txt"):
    raw_data_paths = glob(raw_data)
    acc_files = [file for file in raw_data_paths if "acc" in file]
    gyro_files = [file for file in raw_data_paths if "gyro" in file]
    label_files = [file for file in raw_data_paths if "labels" in file]

    raw_acc_columns=['acc_X','acc_Y','acc_Z']
    raw_gyro_columns=['gyro_X','gyro_Y','gyro_Z']
    raw_labels_columns=['experiment_number_ID','user_number_ID','activity_number_ID','label_start_point','label_end_point']
    acc_df = read_raw_data(acc_files, raw_acc_columns)
    gyro_df = read_raw_data(gyro_files, raw_gyro_columns)
    label_df = read_raw_data(label_files, raw_labels_columns)
    return acc_df, gyro_df, label_df

def read_raw_data(files, columns):
    extra_cols = []
    dfs = [pd.read_csv(f, sep=' ', header=None) for f in files]
    for idx, df in enumerate(dfs):
        filename = files[idx].split("/")[-1]
        ints = list(map(int, re.findall(r'\d+', filename)))
        if len(ints)>0:
            df["experiment_number_ID"] = ints[0]
            df["user_number_ID"] = ints[1]
            df['timestamp'] = df.index
            extra_cols = ["experiment_number_ID", "user_number_ID", "timestamp"]
        dfs[idx] = df
    if len(extra_cols)>0:
        columns+=extra_cols
    df  = pd.concat(dfs, ignore_index=True)
    df.columns = columns 
    return df

def load_signal(label_df, acc_df, gyro_df, user_number_ID=1, activity_number_ID=6, experiment_number_ID=1):
    user_df = label_df[label_df.user_number_ID==user_number_ID]
    user_df = user_df[user_df.activity_number_ID==activity_number_ID]
    user_df = user_df[user_df.experiment_number_ID==experiment_number_ID]
    
    acc_df = acc_df[acc_df.experiment_number_ID.isin(user_df.experiment_number_ID) & acc_df.user_number_ID.isin(user_df.user_number_ID)]
    acc_df = acc_df[acc_df.timestamp.between(user_df.label_start_point.values[0], user_df.label_end_point.values[0])]
    gyro_df = gyro_df[gyro_df.experiment_number_ID.isin(user_df.experiment_number_ID) & gyro_df.user_number_ID.isin(user_df.user_number_ID)]
    gyro_df = gyro_df[gyro_df.timestamp.between(user_df.label_start_point.values[0], user_df.label_end_point.values[0])]
    return acc_df, gyro_df


def visualize_triaxial_signals(acc_df, gyro_df, sampling_freq=50):
    
    len_df=len(acc_df)
    time=[1/float(sampling_freq) *j for j in range(len_df)]

    plt.figure(figsize=(18,5))
    plt.plot(time, acc_df.acc_X, label='acc_X', color='red')
    plt.plot(time, acc_df.acc_Y, label='acc_Y', color='green')
    plt.plot(time, acc_df.acc_Z, label='acc_Z', color='blue')
    plt.xlabel('Time in seconds (s)')
    plt.ylabel('Acceleration in 1g')
    plt.show()
    
    plt.figure(figsize=(18,5))
    plt.plot(time, gyro_df.gyro_X, label='gyro_X', color='red')
    plt.plot(time, gyro_df.gyro_Y, label='gyro_Y', color='green')
    plt.plot(time, gyro_df.gyro_Z, label='gyro_Z', color='blue')
    plt.xlabel('Time in seconds (s)') 
    plt.ylabel('Angular Velocity in radian per second [rad/s]')
    plt.show()