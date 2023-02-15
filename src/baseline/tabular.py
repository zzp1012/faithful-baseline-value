import os
import copy
import math
from tqdm import tqdm

# import ML libs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# import internal libs
from config import logger, DEVIATION_NUM, LOW_ORDER, EXTREME_SMALL_VAL
from tools.utils import compute_vS


def learn_baseline(args, model, X, y, X_train):
    """Training the baseline values. And get multiple related plots.

    Args:
        args (dict): set containing all program arguments
        model (nn.module): the trained model using X implemented with pytorch. Model.to(device)
        X (torch 2d Tensor): sampled dataset.
        y (torch 1d Tensor): label corresponding to the sampled dataset.
        X_train (tensor.to(deivice)): the entire dataset containing all features. (# of samples, 1, IMG_SIZE, IMG_SIZE).
    
    Returns:
        None
    """

    ## define some reused constants
    num_of_features = X.size(-1)
    low_orders = np.arange(num_of_features)[:math.floor(num_of_features * LOW_ORDER)] # 0 ~ floor(# of features * 0.5)

    ## set the min and max threshold for the baseline value
    train_min = X_train.mean(dim=0, keepdim = True) - DEVIATION_NUM * X_train.std(dim = 0, keepdim = True) # (1, num_of_features)
    train_max = X_train.mean(dim=0, keepdim = True) + DEVIATION_NUM * X_train.std(dim = 0, keepdim = True) # (1, num_of_features)

    ## initialise the baseline values
    if args.init == "mean":
        baselines = torch.reshape(copy.deepcopy(X_train.detach().mean(dim = 0)), (1, num_of_features)).to(args.device).requires_grad_(True)
    elif args.init == "zero":
        baselines = torch.zeros((1, num_of_features), requires_grad = True).to(args.device)

    baselines = nn.Parameter(baselines.float())
    
    # setting the model status
    model.eval()
    model.requires_grad = False # dosen't change parameters in the model.

    ## store the data
    res_global_dict = {
        'loss': [],
        **{'baseline_{}'.format(i): [] for i in range(num_of_features)}, # 2 ~ num_of_features + 1
    }

    # learning the baseline value
    for itr in range(args.itr):
        logger.info("############# Iter: {}".format(itr))
        # define the function of calculate_grad.

        # calculating gradients on dataset level
        logger.info("calculating gradients for each context...")
        loss, grads = calculate_grad(args, baselines, model, X, y, low_orders)

        for metric, val in [("loss", loss)]:
            logger.info("{} ->\t {}".format(metric, val))
            logger.info("-------------------------")

        # store the partial res in dict and csv
        res_global_dict['loss'].append(loss.detach().cpu().numpy())
        for i in range(num_of_features):
            res_global_dict['baseline_{}'.format(i)].append(baselines.flatten().detach().cpu().numpy().ravel()[i])

        # store the res in csv
        res_global_df = pd.DataFrame.from_dict(res_global_dict)
        csv_path = os.path.join(args.save_path, "reference_global.csv")
        res_global_df.to_csv(csv_path, index=False)

        # update the baseline values
        baselines = baselines.data - args.lr * grads
        baselines = torch.max(torch.min(baselines, train_max), train_min).clone().detach().requires_grad_(True).float() # clamp
        baselines = nn.Parameter(baselines)

    np.save(os.path.join(args.save_path, "baseline.npy"), baselines.detach().cpu().numpy())


def calculate_grad(args, baselines, model, X_input, y_input, low_orders):
    """calculate scores and grad over baselines values for any inputs.

    Args:
        args (dict): set containing all program arguments
        baselines (tensor.to(device)): baseline values to be learned, should be (1, num_of_features)
        model (nn.module): the trained model using X implemented with pytorch. Model.to(device)
        X_input (tensor.to(device)): should be (num_of_input, num_of_features) sampled
        y_input (tensor.to(device)): should be (num_of_input, num_of_features) sampled
        low_orders (np.array): the order of Shapley values or marginal benefits to be penalized

    Returns:
        loss scalar
        grads (num_of_features, ) numpy array
    """

    num_of_inputs = X_input.size(0)
    num_of_features = X_input.size(-1)
    num_of_samples = args.sample_num
    num_of_order = len(low_orders)

    # calculate the gradient and shap value for each feature
    batch_id = 0
    batch_start = 0
    input_batch = []
    input_i_batch = []
    batch_num = math.ceil(num_of_features / args.baseline_bs)

    # define softmax layer
    softmax_func = nn.Softmax(dim=-1).to(args.device)

    grads = torch.zeros((batch_num, num_of_features)).to(args.device)
    score = 0

    for i in tqdm(range(num_of_features)):
        if baselines.grad is not None:
            baselines.grad.zero_()
        inputs = []
        inputs_i = []

        # generate contexts
        contexts = torch.Tensor(num_of_samples, num_of_inputs, num_of_features).to(args.device).uniform_(0, 1)
        for m in low_orders:
            # create masks that retain $m$ variables to represent the context $S$ with $|S|=m$
            masks_tmp = (contexts < (m / num_of_features)).int()
            masks_tmp[:, :, i] = 0
            masks_tmp_i = masks_tmp.clone().detach()
            masks_tmp_i[:, :, i] = 1

            # get $x_S$
            inputs.append((X_input * masks_tmp + (1 - masks_tmp) * baselines).view(-1, num_of_features)) # (num_of_samples * num_of_inputs, num_of_features)
            # get $x_{S\cup {i}}$
            inputs_i.append((X_input * masks_tmp_i + (1 - masks_tmp_i) * baselines).view(-1, num_of_features)) # (num_of_samples * num_of_inputs, num_of_features)
            del masks_tmp, masks_tmp_i

        input_batch.append(torch.cat(inputs))
        input_i_batch.append(torch.cat(inputs_i))  # (num_of_order * num_of_samples * num_of_inputs, num_of_features)
        del inputs, inputs_i, contexts

        # only compute outputs of the model for each full batch of input variables
        length = i - batch_start + 1
        if length < args.baseline_bs and i < num_of_features - 1:
            continue

        input_batch = torch.cat(input_batch)  # (length * num_of_order * num_of_samples * num_of_inputs, num_of_features)
        input_i_batch = torch.cat(input_i_batch)  # (length * num_of_order * num_of_samples * num_of_inputs, num_of_features)

        if args.vfunc == "log-odds":
            # log-odds: v(S) = log(p(y^{truth})/(1-p(y^{truth}))), used with args.loss="shapley"

            model = model.double()
            input_batch, input_i_batch = input_batch.double(), input_i_batch.double()
            # creat one hot lbls
            one_hot_lbl = nn.functional.one_hot(y_input).reshape(1, num_of_inputs, -1)  # (1, num_of_inputs, out_dim)

            vS = compute_vS(model,input_batch, one_hot_lbl, softmax_func, length, num_of_order, num_of_samples, num_of_inputs)
            vS_i = compute_vS(model,input_i_batch, one_hot_lbl, softmax_func, length, num_of_order, num_of_samples, num_of_inputs)

            if args.loss == 'shapley':
                shapley = vS_i.mean(dim=2) - vS.mean(dim=2)  # phi_i^m = E_S[v(S\cup {i})-v(S)]: (length, num_of_order, num_of_inputs, 1)
                shapley_detach = shapley.clone().detach()
                loss = torch.sum(shapley * torch.sign(shapley_detach), dim=(0, 1)).mean() # loss = E_x[\sum_{m}\sum_{i} |\phi_i^m|]
            elif args.loss == 'marginal':
                deltaV = vS_i - vS  # Delta v_i(S): (length, num_of_order, num_of_samples, num_of_inputs, 1)
                deltaV_detach = deltaV.clone().detach()
                loss = torch.sum((deltaV * torch.sign(deltaV_detach)).mean(2), dim=(0, 1)).mean() # loss = E_x[\sum_{m}\sum_{i}E_S[|\Delta v_i(S)|]]


        elif args.vfunc == "l1":
            # l1: |\Delta v_i(S)| = \Vert h(x_{S∪{i}})−h(x_S) \Vert_1, used with args.loss="marginal"

            feature = model(input_batch, return_feature=True)  # h(x_S): (length * num_of_order * num_of_samples * num_of_inputs, out_dim)
            feature = feature.reshape(length, num_of_order, num_of_samples, num_of_inputs,-1)  # (length, num_of_order, num_of_samples, num_of_inputs, out_dim)

            feature_i = model(input_i_batch, return_feature=True)  # h(x_{S\cup {i}})
            feature_i = feature_i.reshape(length, num_of_order, num_of_samples, num_of_inputs, -1)

            deltaV_abs = torch.norm(feature_i - feature, p=1, dim=-1, keepdim=True)  # |\Delta v_i(S)|: (length, num_of_order, num_of_samples, num_of_inputs, 1) use l1-norm
            loss = torch.sum(deltaV_abs.mean(2), dim=(0, 1)).mean() # loss = E_x[\sum_{i\in N}\sum_{m}E_{S,|S|=m}[|\Delta v_i(S)|]]
            del deltaV_abs

        del input_batch, input_i_batch

        loss.backward()
        grads[batch_id] = baselines.grad.clone()
        score += loss.clone().detach()

        del loss

        batch_start += length
        batch_id += 1
        input_batch = []
        input_i_batch = []

    # calculate the overall grad
    grads = torch.sum(grads, dim=0)
    return score, grads
