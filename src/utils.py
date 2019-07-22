import os
import logging
import logging.handlers
logger = logging.getLogger()

import torch
import torch.nn as nn

def save_model(model, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, name), 'wb') as f:
        torch.save(model.state_dict(), f)

def load_model(model, path):
    with open(path, 'rb') as f:
        model.load_state_dict(torch.load(f))

def init_logger(log_file=""):
    logger = logging.getLogger()
    fmt = '[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)s] %(message)s'
    formatter = logging.Formatter(fmt)
    if log_file is not "":
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

def load_dict(path):
    logger.info('load dictionary : %s' % path)
    tags = ['PA', 'PE', 'PD', 'PH', 'PG', 'PB', 'PK',
            'NA', 'NB', 'NJ', 'NH', 'PF', 'NI', 'NC',
            'NG', 'NE', 'ND', 'NN', 'NK', 'NL', 'PC']
    h = {}
    for i in range(21):
        h[tags[i]] = i

    file = open(path, 'r', encoding='gbk')
    dictionary = {}
    for line in file:
        items = line.strip().split(',')
        if (len(items) > 12):
            continue
        dictionary[items[0]] = (h[items[4].strip()], int(items[5]))
    logger.info('dict size : %d' % len(dictionary))
    return dictionary

def load_pop_dict(path):
    logger.info('load popular dictionary : %s' % path)

    file = open(path, 'r')
    dictionary = {}
    for line in file:
        items = line.strip().split()
        if (len(items) < 2):
            continue
        dictionary[items[0]] = float(items[1])
    logger.info('dict size : %d' % len(dictionary))
    return dictionary

def ffnn(num_layers, input_size, output_size, hidden_size):
    ffnn = nn.ModuleList()
    if num_layers == 1:
        ffnn.append(nn.Linear(input_size, output_size))
    else:
        ffnn.append(
            nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh()))
        for i in range(num_layers - 2):
            ffnn.append(
                nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh()))
        ffnn.append(nn.Linear(hidden_size, output_size))
    return ffnn