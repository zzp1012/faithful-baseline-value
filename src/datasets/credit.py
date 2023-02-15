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
from tools.utils import label_encode, recognize_features_type, set_discrete_continuous, stratified_split
from config import logger, USED_SAMPLE_NUM, SPLIT_RATIO


def load_credit(args):
    """prepare the credit dataset including shuffle and splitting.
    credit dataset(binary class), detailed description could be found in https://archive.ics.uci.edu/ml/datasets/South+German+Credit+%28UPDATE%29

    Args:
        args (dict): set containing all program arguments.
    
    Returns:
        train set, test set and the whole X.
    """
    # Read Dataset
    df = pd.read_csv(os.path.join(args.data_path, 'german_credit.csv'), delimiter=',')

    # Features Categorization
    columns = df.columns
    class_name = 'default'

    type_features, _ = recognize_features_type(df)

    discrete = ['installment_as_income_perc', 'present_res_since', 'credits_this_bank', 'people_under_maintenance']
    discrete, _ = set_discrete_continuous(columns, type_features, class_name, discrete, continuous=None)

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