import matplotlib.pyplot as plt
import featureeng
import pandas as pd
from featureeng.math import Scaling


def presentData(data_frame, columns=[], scaling=None):
    """

    :param data_frame:
    :param columns:
    :param scaling: minmax, normalize
    :return:
    """
    if isinstance(data_frame, featureeng.Frame):
        data_frame = data_frame.get_panda_frame()

    if not isinstance(data_frame, pd.DataFrame):
        return

    indices = range(len(data_frame.index))
    plt.title('Chart')
    for column in columns:
        data = map(float, list(data_frame[column]))

        if scaling == 'minmax':
            data = Scaling.minMaxScaling(data)

        plt.plot(indices, data)

    plt.legend(columns, loc='upper right')
    plt.show()


def saveChart(data_frame, columns=[], file_name='figure.png', scaling=None):
    indices = range(len(data_frame.index))
    plt.title('Chart')
    for column in columns:
        data = map(float, list(data_frame[column]))

        if scaling == 'minmax':
            data = Scaling.minMaxScaling(data)

        plt.plot(indices, data)

    plt.legend(columns, loc='upper left')
    plt.savefig(file_name)
    plt.close()


