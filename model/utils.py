import logging
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm


def load_dict_npy_data(filename):
    return np.load(filename, allow_pickle=True).item()


def set_environment(seed, benchmark_flags=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # if you are using multi-GPU
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # speed up the conv operation
        torch.backends.cudnn.benchmark = benchmark_flags


def read_properties(properties_file):
    properties = {}
    with open(properties_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            key, value = line.split("=")
            properties[key.strip()] = value.strip()
    return properties


def save_checkpoints(model, optimizer, epoch, filepath):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    with open(filepath, "wb") as f:
        pickle.dump(state, f)


def load_checkpoints(filepath, model, optimizer):
    with open(filepath, "rb") as f:
        state = pickle.load(f)
    epoch = state["epoch"]
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return epoch, model, optimizer


def config_logger(logfile):
    # create a logging
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    # config the filehandler
    handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)

    # config the output format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # config the output of console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


class Trainer:
    def __init__(self, dataloader, model, loss, optimizer, scheduler, device, logger):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss = loss
        self.logger = logger

    def train(self, epoch):
        self.model.train()
        losses = 0
        # set print bar
        for item in tqdm(self.dataloader, total=len(self.dataloader)):
            graphs = [graph.to(self.device) for graph in item[0][0]]
            labels = item[0][1][0]
            labels = torch.tensor(labels, dtype=torch.float32)

            self.model.zero_grad()
            predictions = self.model(graphs)

            loss = self.loss(labels, predictions.to('cpu'))
            losses += loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        gpu_mem_alloc = (
            torch.cuda.max_memory_allocated() / 1000000
            if torch.cuda.is_available()
            else 0
        )
        self.logger.info(
            "Epoch: {:03d} | Train Loss: {:.3f} | lr = {:.20f} | GPU occupy: {:.6f} MiB".format(
                epoch + 1,
                loss.item(),
                self.optimizer.state_dict()["param_groups"][0]["lr"],
                gpu_mem_alloc,
            )
        )
        avg_epoch_loss = losses / (len(self.dataloader) * 31)
        # print(labels)
        # print(predictions)
        return avg_epoch_loss, predictions


class Tester:
    def __init__(self, model, loss, dataloader, device, logger):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.loss = loss
        self.logger = logger

    def test(self):
        self.model.eval()
        losses = 0
        results = []
        for item in tqdm(self.dataloader, total=len(self.dataloader)):
            graphs = [graph.to(self.device) for graph in item[0][0]]
            labels = item[0][1][0]
            labels = torch.tensor(labels, dtype=torch.float32)

            predictions = self.model(graphs)
            loss = self.loss(labels, predictions.to('cpu'))
            losses += loss.item()
            results.append(predictions.tolist())

        gpu_mem_alloc = (
            torch.cuda.max_memory_allocated() / 1000000
            if torch.cuda.is_available()
            else 0
        )
        self.logger.info(
            "Test Loss: {:.6f} | GPU occupy: {:.6f} MiB".format(
                loss.item(), gpu_mem_alloc
            )
        )
        avg_epoch_loss = losses / (len(self.dataloader) * 31)
        return avg_epoch_loss, results


def draw_curve(epochs, title, losses):

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    sns.lineplot(x= [x for x in range(1,epochs+1)],y = losses)
    plt.legend(title)
    plt.show()