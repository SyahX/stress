import data as Data
import json
import argparse
import logging
import logging.handlers

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="./conf.json")
parser.add_argument("-gpu", "--gpu", type=str, default="-1")
parser.add_argument("-sv", "--save", type=str, default="./save")
parser.add_argument("-ckpt", "--checkpoint", type=str, default=None)
parser.add_argument("-test", "--test_only", type=bool, default=False)
parser.add_argument("-log", "--log", type=str, default="./log")
parser.add_argument("-dict", "--dict", type=str, default="./dict")
args = parser.parse_args()

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

logger = init_logger(args.log)
logger.setLevel(logging.INFO)

def main():
    config = json.load(open(args.config, "r"))
    data = Data.Data(config)

if __name__ == '__main__':
    main()