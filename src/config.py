import os, logging
from datetime import datetime
from six.moves import urllib

# make the downloader for mnist work
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# current date
DATE = str(datetime.now().month).zfill(2)+str(datetime.now().day).zfill(2)
# current time
MOMENT = str(datetime.now().hour).zfill(2)+str(datetime.now().minute).zfill(2)+str(datetime.now().second).zfill(2)

# set the logger
logging.basicConfig(filename = os.path.join(".", "logfile-{}-{}.txt".format(DATE, MOMENT)), 
                    filemode = "w+", 
                    format = '%(name)-12s: %(levelname)-8s %(message)s', 
                    datefmt = "%H:%M:%S", 
                    level = logging.INFO)
# define a Handler which writes DEBUG messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# tell the handler to use this format
console.setFormatter(logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s'))
# add the handler to the root logger
logging.getLogger().addHandler(console)
logger = logging.getLogger("tabular")

## set GLOBAL constants
DATASETS = ["credit", "census"]

# supporting initial methods
INITS = ["random", "zero", "mean"]

# supporting loss functions
LOSSES = ["shapley", "marginal"]

# supporting v functions
# log-odds: v(S) = log(p(y^{truth})/(1-p(y^{truth}))), used with args.loss="shapley"
# l1: |\Delta v_i(S)| = \Vert h(x_{S∪{i}})−h(x_S) \Vert_1, used with args.loss="marginal"
VFUNCS = ["log-odds", "l1"]

# the number of sampling used
USED_SAMPLE_NUM = {
    "credit": 500,
    "census": 1000
} # must be a even number

# splitting ratio
SPLIT_RATIO = {
    "credit": 0.2,
    "census": 0.2
}


# when plot multiple lines, the maximum number of lines within one figure.
MAX_NUM_LINE = 25 

# the actuall numbe of lines within one graph.
ACTUAL_NUM_LINE = 10 

# parameter for clamp, train_max = X_train.mean + a * X_train.std, here we aims to set "a"
DEVIATION_NUM = 0.3

# the order
LOW_ORDER = 0.5

# Small value
EXTREME_SMALL_VAL = 0.0000000000000001