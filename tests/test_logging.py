from ecnet import Server


def test_init(log_level, log_dir):

    sv = Server(log_dir=log_dir, log_level=log_level)
    sv._logger.log(
        'crit',
        'LOGGING INIT | log_level: {} | log_dir: {}'.format(
            log_level, log_dir
        ),
        call_loc='UNIT TESTING'
    )
    sv._logger.log('debug', 'Debug')
    sv._logger.log('info', 'Info')
    sv._logger.log('warn', 'Warning')
    sv._logger.log('error', 'Error')
    sv._logger.log('crit', 'Critical')


def test_set(log_level, log_dir):

    sv = Server()
    sv._logger.log(
        'crit',
        'LOGGING SET | log_level: {} | log_dir: {}'.format(
            log_level, log_dir
        ),
        call_loc='UNIT TESTING'
    )
    sv._logger.log('debug', 'Debug')
    sv._logger.log('info', 'Info')
    sv._logger.log('warn', 'Warning')
    sv._logger.log('error', 'Error')
    sv._logger.log('crit', 'Critical')
    sv.log_level = log_level
    sv.log_dir = log_dir
    sv._logger.log('debug', 'Debug')
    sv._logger.log('info', 'Info')
    sv._logger.log('warn', 'Warning')
    sv._logger.log('error', 'Error')
    sv._logger.log('crit', 'Critical')


if __name__ == '__main__':

    levels = ['debug', 'info', 'warn', 'error', 'crit']
    dirs = [None, './log_test/']
    for l in levels:
        for d in dirs:
            test_init(l, d)
            test_set(l, d)
