import os

# import ML libs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# import internal libs
from models import MLP
from tools.plot import plot_curves, plot_prob
from config import logger, DATE, MOMENT


def train_model(args, X_train, y_train, X_test, y_test, train_loader):
    """prepare the model the later usage. If the model haven't been trained, we will firstly 
    train the model. If the model has been trained, we will certainly load the model.

    Args:
        args (dict): set containing all program arguments.
        X_train, y_train, X_test, y_test (tensor.to(deivice)): splitted dataset.
        train_loader (nn.dataloader) used for training model

    Returns:
        return the fitted model or trained model.
    """    
    # get the net
    in_dims = X_train.size(-1)
    out_dims = len(set(y_train.detach().cpu().numpy()))
    if args.dataset in ["census", "credit"]:
        net = MLP(in_dim=in_dims, hidd_dim=100, out_dim=out_dims)
    else:
        raise NotImplementedError ("{} dataset is not implemented.".format(args.dataset))
    net = net.float().to(args.device)
    
    # define loss function
    criterion = nn.CrossEntropyLoss().to(args.device)
    
    model_name = "dataset={}-model={}-epoch={}-seed={}-bs={}-lr={}-logspace={}.pt".format(args.dataset,"mlp", args.epoch, args.model_seed, args.batch_size, args.train_lr, args.logspace)
    if model_name in os.listdir(args.model_path):
        logger.info("The {} has existed in model path '{}'. Load pretrained model.".format(model_name, args.model_path))
        net.load_state_dict(torch.load(os.path.join(args.model_path, model_name)))  
        
        # evaluate the performance of the model
        net.eval()
        with torch.no_grad():
            logits = net(X_test)
            loss = criterion(logits, y_test)
            targets = torch.argmax(logits, dim = -1)
            acc = torch.sum(targets == y_test).item() / len(y_test)

        logger.info('On test set - \t Loss: {:.6f} \t Acc: {:.9f}'.format(loss.item(), acc))

        return net
    else: 
        logger.info("The {} doen't exist in model path '{}'. Train a model with new settings.".format(model_name, args.model_path))
        if args.model_seed != args.seed:
            logger.error("argument model_seed should be as same as argument seed to train a new model with new settings. Terminate program with code -3.")
            exit(-3)

        # define the optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr = args.train_lr)
        # set the decay of learning rate
        if args.logspace != 0:
            logspace_lr = np.logspace(np.log10(args.train_lr), np.log10(args.train_lr) - args.logspace, args.epoch)

        # define the train_csv
        learning_csv = os.path.join(args.save_path, "learning.csv")
        # define the res dict
        res_dict = {
            'train-loss': [], 'train-acc': [], 'test-loss': [], "test-acc": []
        }
        
        # starts training
        for epoch in range(args.epoch + 1): # to test the acc for the last time
            # eval on test set
            net.eval()
            with torch.no_grad():
                # eval on train
                train_log_probs = net(X_train)
                train_loss = criterion(train_log_probs, y_train)
                train_targets = torch.argmax(train_log_probs, dim = -1)
                train_acc = torch.sum(train_targets == y_train).item() / len(y_train)

                # eval on test
                test_log_probs = net(X_test)
                test_loss = criterion(test_log_probs, y_test)
                test_targets = torch.argmax(test_log_probs, dim = -1)
                test_acc = torch.sum(test_targets == y_test).item() / len(y_test)

            # save the res in dict
            res_dict['train-loss'].append(train_loss.item()); res_dict["train-acc"].append(train_acc)
            res_dict['test-loss'].append(test_loss.item()); res_dict["test-acc"].append(test_acc);
            # store the res in csv
            pd.DataFrame.from_dict(res_dict).to_csv(learning_csv, index = False)
        
            # show loss
            if epoch % 10 == 0 or epoch == args.epoch:
                logger.info('On train set - Epoch: {} \t Loss: {:.6f} \t Acc: {:.9f}'.format(epoch, train_loss.item(), train_acc))
                logger.info('On test set - Epoch: {} \t Loss: {:.6f} \t Acc: {:.9f}'.format(epoch, test_loss.item(), test_acc))

                if epoch % 100 == 0 or epoch == args.epoch - 1:
                    # draw the curves
                    plot_curves(args, res_dict)
            
            if epoch >= args.epoch:
                break
                
            # set the lr
            if args.logspace != 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = logspace_lr[epoch]
            for step, (batch_X, batch_y) in enumerate(train_loader):
                batch_X.to(args.device); batch_y.to(args.device)
                net.train()
                net.zero_grad()
                log_probs = net(batch_X)
                loss = criterion(log_probs, batch_y)
                loss.backward()
                optimizer.step()

        # save the model
        torch.save(net.cpu().state_dict(), os.path.join(args.model_path, model_name))      
        logger.info("The {} has trained and saved in model path '{}'. Terminate program with code 0".format(model_name, args.model_path))
        
        # move logfile.txt to save_root.
        os.system("mv logfile-{}-{}.txt {}".format(DATE, MOMENT, args.save_path))
        exit(0)
