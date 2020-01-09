import unittest

from ecnet.utils.logging import logger

from models.test_mlp import TestMLP
from models.test_pretrained import TestPretrained
from tasks.test_limit_inputs import TestLimit
from tasks.test_tune_hyperparameters import TestTune
from tools.test_database import TestDatabase
from tools.test_project import TestUseProject
from utils.test_data_utils import TestDataUtils
from utils.test_error_utils import TestErrors
from utils.test_server_utils import TestServerUtils
from server.test_server import TestServer


if __name__ == '__main__':

    logger.stream_level = 'debug'
    unittest.main()
