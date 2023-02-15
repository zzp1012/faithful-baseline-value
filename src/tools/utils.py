import os
import random
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

from config import logger
from config import EXTREME_SMALL_VAL


def set_seed(seed = 0):
    """set the random seed for multiple packages.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_device(device):
    """set GPU device.
    """
    if torch.cuda.is_available():
        if device >= torch.cuda.device_count():
            logger.error("CUDA error, invalid device ordinal")
            exit(1)
    else:
        logger.error("Plz choose other machine with GPU to run the program")
        exit(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    device = torch.device("cuda:" + str(device))
    logger.info(device) 
    return device


def model_test(net):
    """output the model infomation.
    """
    total_params = 0
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.cpu().numpy().shape)
    logger.info("Total number of params {}".format(total_params))
    logger.info("Total layers {}".format(len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters())))))


def recognize_features_type(df):    
    integer_features = list(df.select_dtypes(include=['int64']).columns)
    double_features = list(df.select_dtypes(include=['float64']).columns)
    string_features = list(df.select_dtypes(include=['object']).columns)
    type_features = {
        'integer': integer_features,
        'double': double_features,
        'string': string_features,
    }
    features_type = dict()
    for col in integer_features:
        features_type[col] = 'integer'
    for col in double_features:
        features_type[col] = 'double'
    for col in string_features:
        features_type[col] = 'string'
        
    return type_features, features_type


def set_discrete_continuous(features, type_features, class_name, discrete=None, continuous=None): 
    if discrete is None and continuous is None:
        discrete = type_features['string']
        continuous = type_features['integer'] + type_features['double']
        
    if discrete is None and continuous is not None:
        discrete = [f for f in features if f not in continuous]
        continuous = list(set(continuous + type_features['integer'] + type_features['double']))
        
    if continuous is None and discrete is not None:
        continuous = [f for f in features if f not in discrete and (f in type_features['integer'] or f in type_features['double'])]
        discrete = list(set(discrete + type_features['string']))
    
    discrete = [f for f in discrete if f != class_name] + [class_name]
    continuous = [f for f in continuous if f != class_name]
    return discrete, continuous


def label_encode(df, columns):
    df_le = df.copy(deep=True)
    label_encoder = dict()
    for col in columns:
        le = LabelEncoder()
        df_le[col] = le.fit_transform(df_le[col])
        label_encoder[col] = le
    return df_le, label_encoder


def stratified_split(y, train_ratio):
    """split the imbalance dataset into balanced 
    """
    def split_class(y, label, train_ratio):
        indices = np.flatnonzero(y == label)
        n_train = int(y.size*train_ratio/len(np.unique(y)))
        train_index = indices[:n_train]
        test_index = indices[n_train:]
        return (train_index, test_index)
        
    idx = [split_class(y, label, train_ratio) for label in np.unique(y)]
    train_index = np.concatenate([train for train, _ in idx])
    test_index = np.concatenate([test for _, test in idx])
    return train_index, test_index

def compute_vS(model,x,y,softmax_func, length, num_of_order, num_of_samples, num_of_inputs):
    logits = model(x)  # (length * num_of_order * num_of_samples * num_of_inputs, out_dim)
    logits = logits.reshape(length, num_of_order, num_of_samples, num_of_inputs,-1)  # (length, num_of_order, num_of_samples, num_of_inputs, out_dim)
    prob = softmax_func(logits)  # p: (length, num_of_order, num_of_samples, num_of_inputs, out_dim)
    prob = (y * prob).sum(dim=-1, keepdim=True)  # p(y^{truth}|x_S): (length, num_of_order, num_of_samples, num_of_inputs, 1)

    prob[prob >= 1 - EXTREME_SMALL_VAL] = 1 - EXTREME_SMALL_VAL
    prob[prob <= 0 + EXTREME_SMALL_VAL] = 0 + EXTREME_SMALL_VAL
    vS = torch.log(prob / (1 - prob))  # v(S): (length, num_of_order, num_of_samples, num_of_inputs, 1)
    return vS
