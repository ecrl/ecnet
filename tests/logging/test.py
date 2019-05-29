from ecnet import Server
from ecnet.utils.logging import logger


def stream_logging(s_level):

    logger.stream_level = s_level
    logger.log('debug', 'Debug message')
    logger.log('info', 'Info message')
    logger.log('warn', 'Warning message')
    logger.log('error', 'Error message')
    logger.log('crit', 'Critical message')


def file_logging(f_level, log_dir='logs'):

    logger.file_level = f_level
    logger.stream_level = f_level
    logger.log_dir = log_dir
    logger.log('debug', 'Debug message')
    logger.log('info', 'Info message')
    logger.log('warn', 'Warning message')
    logger.log('error', 'Error message')
    logger.log('crit', 'Critical message')


if __name__ == '__main__':

    levels = ['debug', 'info', 'warn', 'error', 'crit', 'disable']
    for l in levels:
        stream_logging(l)
    for l in levels:
        file_logging(l)
    file_logging('debug', 'new_logs')
