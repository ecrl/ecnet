from ecnet import Server


def no_logging():

    sv = Server(log_progress=False)
    sv.import_data('cn_model_v1.0.csv')

if __name__ == '__main__':

    no_logging()
