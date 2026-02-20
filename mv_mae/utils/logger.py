import logging
import os
import sys

def setup_logger(save_dir, log_filename="training.log"):
    """
    Sets up a logger that outputs to both the console and a log file
    """
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, log_filename)

    logger = logging.getLogger("MV_MAE LOGGER")
    logger.setLevel(logging.INFO)

    #clear preious handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    #file handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    #console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

    
