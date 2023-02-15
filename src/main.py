import os
import logging
import argparse
import torch
import json

# import internal libs
import datasets
import baseline
from tools.train import train_model
from tools.utils import set_seed, set_device, model_test
from config import logger, DATE, MOMENT, DATASETS, INITS, LOSSES, VFUNCS


def add_args():
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    logger.info("-------Parsing program arguments--------")
    parser = argparse.ArgumentParser(description = "Code for reference-value experiment (binary classfication)")

    ## the basic setting of exp
    parser.add_argument('--device', default = 1, type = int,
                        help = "set the device.")
    parser.add_argument("--seed", default = 1, type = int,
                        help = "set the random seed.")
    parser.add_argument("--dataset", default = "census", type = str,
                        choices = DATASETS,
                        help = "set the dataset used.")

    # set the path for data
    parser.add_argument('--data_path', default = '../data/', type = str, 
                        help = "path for dataset.")
    # set the model path.
    parser.add_argument("--model_path", default = "../models", type = str, 
                        help = 'the path of pretrained model.')
    # path for saved file
    parser.add_argument("--save_root", default = "../results", type = str,
                        help = 'the path of saved fig.')
    
    ## the setting for training model
    # set the model seed
    parser.add_argument("--model_seed", default = 2, type = int,
                        help = "set the seed used for training model.")
    # set the batch size for training
    parser.add_argument('--batch_size', default = 512, type = int,
                        help="set the batch size for training.")
    # set the learning rate for training
    parser.add_argument('--train_lr', default = 0.01, type = float,
                        help="set the learning rate for training.")
    # set the decay of learning rate
    parser.add_argument("--logspace", default = 1, type=int,
                        help='the decay of learning rate.')
    # set the number of epochs for training model.
    parser.add_argument("--epoch", default = 300, type = int,
                        help = 'the number of iterations for training model.')

    ## the setting for baseline value experiment
    # set the initial baseline values (mean, zero, random)
    parser.add_argument('--init', default = 'zero', type = str,
                        choices = INITS,
                        help = "set the method to initial the baseline values [zero|mean]")
    parser.add_argument("--loss", default='shapley', type=str,
                        choices = LOSSES,
                        help = "set the loss function used to learn baseline values [shapley|marginal].") # shap, v
    parser.add_argument('--vfunc', default = 'log-odds', type = str,
                        choices = VFUNCS,
                        help = "set the v(S) for calculating shapley value [log-odds|l1]")
    # set the learning rate for baseline value
    parser.add_argument("--lr", default = 0.01, type = float,
                        help = "set the learning rate for learning baseline values.")
    # set the number of epochs for training model.
    parser.add_argument("--itr", default = 2, type = int,
                        help = 'the number of iterations for learning baseline values.')
    # set the batch size for baseline value
    parser.add_argument('--baseline_bs', default=1, type=int,
                        help="set the batch size for baseline value.")
    # set the number of times of sampling
    parser.add_argument("--sample_num", default = 100, type = int,
                        help = 'the number of samples when using contexts.')
    args = parser.parse_args()

    if (args.vfunc == "l1" ) and (args.loss != "marginal"):
        logger.warning("for v(S) = l1 norm, the loss is forced to be marginal")
        args.loss = "marginal"

    if not os.path.exists(args.data_path):
        logger.error("The data path doesn't exist. Terminate program with code -1.")
        exit(-1)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    return args


def main():
    # get the args.
    args = add_args()

    # set the save_path
    exp_name = "-".join([DATE, MOMENT, args.dataset, "lr={}".format(args.lr), "itr={}".format(args.itr), "seed={}".format(args.seed),
                        "init={}".format(args.init), "loss={}".format(args.loss), "vfunc={}".format(args.vfunc), "sample={}".format(args.sample_num)])
    args.save_path = os.path.join(args.save_root, exp_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # set the seed
    set_seed(args.seed)

    # set the device
    args.device = set_device(args.device)

    # get the dataset
    logger.setLevel(logging.INFO)
    logger.info("-----------preparing dataset-----------")
    logger.info("dataset - {}".format(args.dataset))
    if args.dataset == "credit":
        X_train, y_train, X_test, y_test, X, y, train_loader = datasets.load_credit(args)
    elif args.dataset == "census":
        X_train, y_train, X_test, y_test, X, y, train_loader = datasets.load_census(args)
    else:
        raise NotImplementedError ("{} dataset is not implemented.".format(args.dataset))

    # training the model
    logger.info("------------preparing model------------")
    model = train_model(args, X_train, y_train, X_test, y_test, train_loader)
    model_test(model)
    
    # learning the baseline values
    logger.info("-------learning baseline values--------")
    baseline.tabular.learn_baseline(args, model, X, y, X_train)

    # move logfile.txt to save_root.
    os.system("mv logfile-{}-{}.txt {}".format(DATE, MOMENT, args.save_path))


if __name__ == "__main__":
    main()
