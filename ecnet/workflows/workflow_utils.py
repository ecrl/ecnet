from ecnet.utils.data_utils import DataFrame
from ecnet.utils.server_utils import default_config, train_model
from ecnet.tasks.limit_inputs import limit_rforest

from multiprocessing import Pool, set_start_method
from os import name


def prop_range_from_split(db_name: str, data_split: list):
    ''' prop_range_from_split: creates learning, validation, test subsets, each
    with proportionally equal number of compounds based on experimental value
    range

    Args:
        db_name (str): name/location of ECNet-formatted database
        data_split (list): [learn %, valid %, test %] for all supplied data
    '''

    df = DataFrame(db_name)
    for mol in df.data_points:
        mol.assignment = 'L'
        mol.TARGET = float(mol.TARGET)
    df.data_points = sorted(df.data_points, key=lambda m: m.TARGET,
                            reverse=True)
    num_validate = int(len(df) / (data_split[1] * len(df.data_points)))
    num_test = int(len(df) / (data_split[2] * len(df.data_points)))
    for idx, mol in enumerate(df.data_points):
        if (idx + 1) % num_validate == 0:
            mol.assignment = 'V'
        if (idx + 3) % num_test == 0:
            mol.assignment = 'T'
    if len([m for m in df.data_points if m.assignment == 'V']) == 0:
        raise ValueError('Data split resulted in empty validation set')
    if len([m for m in df.data_points if m.assignment == 'T']) == 0:
        raise ValueError('Data split resulted in empty test set')
    df.data_points = sorted(df.data_points, key=lambda m: m.id)
    df.save(db_name)


def find_optimal_num_inputs(db_name: str, eval_set: str,
                            num_processes: int) -> tuple:
    ''' find_optimal_num_inputs: find the optimal number of input variables,
    return names of variables; optimal number of variables produces lowest
    median absolute error; variables added 25 at a time, according to RFR
    importance score (most-to-least important)

    Args:
        db_name (str): name/location of ECNet-formatted database
        eval_set (str): set to evaluate (`learn`, `valid`, `train`, `test`,
            None (all))
        num_processes (int): number of concurrent processes to run for RFR,
            training

    Returns:
        tuple: ([addition1, error1, ..., additionN, errorN], opt_desc)
    '''

    conf = default_config()
    conf['epochs'] = 300
    df = DataFrame(db_name)
    df.create_sets()
    desc = limit_rforest(df, len(df._input_names), num_processes=num_processes,
                         eval_set=eval_set)
    desc = [d[0] for d in desc]

    errors = []
    if num_processes > 1:
        if name != 'nt':
            set_start_method('spawn', force=True)
        train_pool = Pool(processes=num_processes)

    for d_idx in range(0, len(desc), 10):
        if d_idx >= len(desc) - 1:
            to_use = desc[:]
        else:
            to_use = desc[: d_idx + 1]
        df = DataFrame(db_name)
        df.set_inputs(to_use)
        df.create_sets()
        sets = df.package_sets()

        if num_processes > 1:
            errors.append([d_idx, train_pool.apply_async(
                train_model, [
                    sets, conf, eval_set, 'med_abs_error', False, '_.h5',
                    False, False
                ]
            )])
        else:
            errors.append([d_idx, train_model(
                sets, conf, eval_set, 'med_abs_error', False, '_.h5',
                False, False
            )])

    if num_processes > 1:
        train_pool.close()
        train_pool.join()
        for idx, err in enumerate(errors):
            errors[idx][1] = err[1].get()

    min_error = errors[0][1]
    opt_num_desc = 1
    for err in errors[1:]:
        if err[1] < min_error:
            min_error = err[1]
            opt_num_desc = err[0]

    return (errors, desc[: opt_num_desc])
