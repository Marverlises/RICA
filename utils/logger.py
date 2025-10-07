import logging
import os
import sys
import os.path as op


def setup_logger(name, save_dir, if_train, distributed_rank=0):
    '''
    This function sets up the logger for the project
    the logger contains two handlers:
    1. StreamHandler: prints the log message to the console
    2. FileHandler: saves the log message to a file
    '''
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if not op.exists(save_dir):
        print(f"{save_dir} is not exists, create given directory")
        os.makedirs(save_dir)
    if if_train:
        fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
    else:
        fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger