import sys
import os
sys.path.append('/home/student2/code/stress/src')
import data as Data
import utils as Utils
import model as Model
import json
import argparse
import logging
import random

import torch

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="./conf.json")
parser.add_argument("-gpu", "--gpu", type=str, default="6")
parser.add_argument("-sv", "--save", type=str, default="./save")
parser.add_argument("-ckpt", "--checkpoint", type=str, default=None)
parser.add_argument("-test", "--test_only", type=bool, default=False)
parser.add_argument("-log", "--log", type=str, default="./log")
parser.add_argument("-dict", "--dict", type=str, default="./dict")
args = parser.parse_args()

logger = Utils.init_logger(args.log)
logger.setLevel(logging.INFO)

gpuDevice = torch.device("cuda:0")
cpuDevice = torch.device("cpu")
device = gpuDevice

def batchify(data, bsz, random_shuffle=True):
    nbatch = len(data) // bsz
    if random_shuffle:
        random.shuffle(data)

    data_batched = []
    for i in range(nbatch):
        batch = data[i * bsz: (i + 1) * bsz]
        x = []
        y = []
        for tx, ty in batch:
            x.append(tx)
            y.append(ty)
        data_batched.append((x, y))
    return data_batched

def evaluate(cnet, data):
    cnet.eval()
    acc = 0
    for x, y in data:
        x = torch.tensor([x], device=device, dtype=torch.float)
        y_pred = torch.argmax(cnet(x)).item()
        if (y_pred == y):
            acc += 1
        """
        y_pred = cnet(x).item()
        if (abs(y_pred - y) < 0.5):
            acc += 1
        """
    total = len(data)
    return acc / total, acc, total

def train(cnet, data, config):
    logger.info("Start train")
    optimizer = torch.optim.Adam(cnet.parameters(), lr=config['lr'])
    # optimizer = torch.optim.SGD(cnet.parameters(), lr=config['lr'])
    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.NLLLoss()

    best_acc = 0.0
    for epoch in range(1, config['epoch'] + 1):
        logger.info("-" * 50)
        train_data = batchify(data.train, config['batch_size'])

        cnet.train()
        train_loss = 0.0
        diff = 0.0
        total = len(train_data) * config['batch_size']
        for t, (x, y) in enumerate(train_data):
            x = torch.tensor(x, device=device, dtype=torch.float)
            y = torch.tensor(y, device=device, dtype=torch.long)

            y_pred = cnet(x)
            diff += torch.sum(torch.abs(torch.argmax(y_pred, dim=1) - y)).item()
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if (t + 1) % config['log_step'] == 0:
                logger.info("Epoch: %d | Batch: %d | loss: %.5f" %
                            (epoch, t + 1, train_loss / (t + 1)))
        logger.info("Train loss : %.5f" % (train_loss / len(train_data)))

        hit = float(total - diff)
        logger.info("Eval train : %d / %d = %.4f%%" %
            (hit, total, hit / total * 100))
        acc, hit, total = evaluate(cnet, data.test)
        logger.info("Eval  test : %d / %d = %.4f%%" %
            (hit, total, acc * 100))
        if acc > best_acc:
            best_acc = acc
            logger.info("Best Accuracy: %.4f" % best_acc)
            Utils.save_model(cnet, args.save, "CNet_best.ckpt")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = json.load(open(args.config, "r"))
    logger.info(config)
    data = Data.Data(config)
    cnet = Model.CNet(config, device)
    cnet.to(device)

    acc, hit, total = evaluate(cnet, data.test)
    logger.info("Eval  test : %d / %d = %.4f%%" %
        (hit, total, acc * 100))
    train(cnet, data, config)

if __name__ == '__main__':
    main()