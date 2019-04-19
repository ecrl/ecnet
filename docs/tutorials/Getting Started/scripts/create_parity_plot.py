from ecnet import Server
from ecnet.utils.logging import logger
from ecnet.tools.plotting import ParityPlot


def main():

    logger.stream_level = 'info'
    sv = Server(prj_file='kinetic_viscosity.prj')

    train_exp = []
    train_exp.extend(y for y in sv._sets.learn_y)
    train_exp.extend(y for y in sv._sets.valid_y)
    train_pred = sv.use(dset='train')
    train_errors = sv.errors('rmse', 'r2', dset='train')

    test_exp = sv._sets.test_y
    test_pred = sv.use(dset='test')
    test_errors = sv.errors('rmse', 'r2', dset='test')

    kv_plot = ParityPlot(
        title='Predicted vs. Experimental Kinematic Viscosity',
        x_label='Experimental KV',
        y_label='Predicted KV'
    )
    kv_plot.add_series(
        train_exp,
        train_pred,
        name='Training Set',
        color='blue'
    )
    kv_plot.add_series(
        test_exp,
        test_pred,
        name='Test Set',
        color='red'
    )
    kv_plot.add_error_bars(test_errors['rmse'], label='Test RMSE')
    kv_plot._add_label('Test R-Squared', test_errors['r2'])
    kv_plot._add_label('Train RMSE', train_errors['rmse'])
    kv_plot._add_label('Train R-Squared', train_errors['r2'])
    kv_plot.save('../kv_parity_plot.png')


if __name__ == '__main__':

    main()
