import unittest

from models.test_mlp import TestMLP
from tasks.test_limit_inputs import TestLimit
from tasks.test_remove_outliers import TestRemoveOutliers
from tasks.test_tune_hyperparameters import TestTune
from tools.test_conversions import TestConversions
from tools.test_database import TestDatabase
from tools.test_project import TestUseProject
from utils.test_data_utils import TestDataUtils
from utils.test_error_utils import TestErrors
from utils.test_server_utils import TestServerUtils


if __name__ == '__main__':

    unittest.main()
