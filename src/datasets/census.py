import os
import copy
import random

# import ML libs
import torch
import torch.utils.data as Data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# import internal libs
from tools.utils import label_encode, recognize_features_type, set_discrete_continuous
from config import logger, USED_SAMPLE_NUM, SPLIT_RATIO


def load_census(args):
    """prepare the credit dataset including shuffle and splitting.
    Extraction was done by Barry Becker from the 1994 Census database. Detailed description could be found in http://archive.ics.uci.edu/ml/datasets/Census+Income

    Args:
        args (dict): set containing all program arguments.
    
    Returns:
        train set, test set, the whole X & y and train_loader.
    """
    # Read Dataset
    df = pd.read_csv(os.path.join(args.data_path, 'adult.csv'), delimiter=',', skipinitialspace=True)

    # Remove useless columns
    del df['fnlwgt']
    del df['education-num']

    # Remove Missing Values
    for col in df.columns:
        if '?' in df[col].unique():
            df[col][df[col] == '?'] = df[col].value_counts().index[0]

    # Features Categorization
    columns = df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    df = df[columns]
    class_name = 'class'
    
    type_features, _ = recognize_features_type(df)

    discrete, _ = set_discrete_continuous(columns, type_features, class_name)

    columns_tmp = list(columns)
    columns_tmp.remove(class_name)

    # Dataset Preparation for Scikit Alorithms
    df_le, _ = label_encode(df, discrete)
    X = df_le.loc[:, df_le.columns != class_name].values
    y = df_le[class_name].values

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT_RATIO[args.dataset], random_state=args.seed)

    logger.info("- > train shape: {}; test shape: {}".format(X_train.shape, X_test.shape))
    logger.debug("sample of X - > \n{}".format(X_train[:1]))
    logger.debug("sample of y - > \n{}".format(y_train[:10]))

    # Sample 1000 X and y from train
    idx_lst = list(range(X_train.shape[0]))
    random.shuffle(idx_lst)
    X = X_train[idx_lst][:USED_SAMPLE_NUM[args.dataset]]
    y = y_train[idx_lst][:USED_SAMPLE_NUM[args.dataset]]
    logger.info("- > sampled dataset shape: {}".format(X.shape))
    
    # prepare the data
    X_train, X_test = torch.from_numpy(X_train).float().to(args.device), torch.from_numpy(X_test).float().to(args.device)
    y_train, y_test = torch.from_numpy(y_train).long().to(args.device), torch.from_numpy(y_test).long().to(args.device)
    X, y = torch.from_numpy(X).float().to(args.device), torch.from_numpy(y).long().to(args.device)

    # create dataloader
    train_set = Data.TensorDataset(X_train, y_train)
    train_loader = Data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
    # prepare the data
    return X_train, y_train, X_test,  y_test, X, y, train_loader