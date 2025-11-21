import os 
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs" # make logs folder and save logs
log_filepath = os.path.join(log_dir,"runnning_logs.log") # saving logs
os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(
    level = logging.INFO,
    format = logging_str,

    handlers=[
        logging.FileHandler(log_filepath), # create log file and save logs in it
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("cnnClassifierLogger")