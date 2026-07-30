import logging
from logging import LoggerAdapter, getLogger, DEBUG, WARNING, FileHandler, Formatter, Logger as Log
from enum import IntEnum
import traceback
from custom_logs.custom_formatter import CustomFormatter
from pathlib import Path
from datetime import datetime

def initilaize_logger():
    now = datetime.now()
    now_formated = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    log_path = Path(f"log\\_{now_formated}.log")

    # create log folder if not exist
    if not log_path.parent.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = Logger(log_path)

class lte(IntEnum):
    debug = 1
    info = 2
    warning = 3
    error = 4
    critical = 5


class FilterCsvLines(logging.Filter):
    def filter(self, record):
        record.msg = '"'+record.msg.replace('"', '""')+'"'
        return record

class Logger:
    """
    Logger for print and log. Support for printing log with different colors on console.
    """
    extra = {}
    logger_adapter: LoggerAdapter
    logger: Log
    log_level = 2
    

    @classmethod
    def __init__(self, file_name):
        
        self.file_name = file_name
        
        # Inicialize Loggers
        self.logger = getLogger('logger')

        # Stream handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(CustomFormatter())
        self.logger.addHandler(ch)

        # File Handler
        file_handler = FileHandler(file_name, encoding="utf-8-sig")
        file_handler.setLevel(logging.INFO)
        file_handler.addFilter(FilterCsvLines())
        format = Formatter('%(asctime)s;%(levelname)s;%(filename)s;%(funcName)s;%(lineno)d;%(message)s')
        file_handler.setFormatter(format)
        self.logger.addHandler(file_handler)

        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)  

    def debugging(self, on=False):
        if on:
            for hdlr in self.logger.handlers:
                hdlr.setLevel(logging.DEBUG)
            self.logger.setLevel(logging.DEBUG)
            self.logger_adapter.setLevel(logging.DEBUG)
        else:
            for hdlr in self.logger.handlers:
                hdlr.setLevel(logging.INFO)
            self.logger.setLevel(logging.INFO)
            self.logger_adapter.setLevel(logging.INFO)


def log(msg, logType=lte.info, **kwargs):
    msg_s = str(msg)

    # if len(kwargs) != 0:
    #     Logger.change_parameters(**kwargs)
    
    log_stacklevel = Logger.log_level
    if logType == lte.debug:
        Logger.logger.debug(msg_s, stacklevel = log_stacklevel)
    elif logType == lte.info:
        Logger.logger.info(msg_s, stacklevel = log_stacklevel)
    elif logType == lte.warning:
        if isinstance(msg, Exception):
            try:
                Logger.logger.warning(traceback.format_exc(), stacklevel = log_stacklevel)
            except Exception:
                Logger.logger.warning(msg_s, stacklevel = log_stacklevel)
        else:
            Logger.logger.warning(msg_s, stacklevel = log_stacklevel)
    elif logType == lte.error:
        if isinstance(msg, Exception):
            try:
                Logger.logger.error(traceback.format_exc(), stacklevel = log_stacklevel)
            except Exception:
                Logger.logger.error(msg_s, stacklevel = log_stacklevel)
        else:
            Logger.logger.error(msg_s, stacklevel = log_stacklevel)
    elif logType == lte.critical:
        if isinstance(msg, Exception):
            try:
                Logger.logger.critical(traceback.format_exc(), stacklevel = log_stacklevel)
            except Exception:
                Logger.logger.critical(msg_s, stacklevel = log_stacklevel)
        else:
            Logger.logger.critical(msg_s, stacklevel = log_stacklevel)