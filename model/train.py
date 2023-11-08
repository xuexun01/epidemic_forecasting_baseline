import argparse

import numpy as np
import pandas as pd
from dataloader import CCDataset
from torch.utils.data import DataLoader
from utils import *


parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--learn_rate", type=float, default=0.001)
parser.add_argument("--weight_decay", type=float, default=0.1)
args = parser.parse_args()

# initialize
logger = config_logger(logfile="./output/run.log")
logger.info("successfully configure the logger...")
logger.info("initialize the environment...")
set_environment(seed=144273)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = read_properties("../config.properties")

dates = pd.date_range(start="2021-01-01", end="2021-04-15")
dates = list(dates.strftime("%Y_%m_%d"))