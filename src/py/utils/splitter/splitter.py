"""
Created on Thu March 03 17:09:39 2022
@ Author: Ma'ruf Nurrochman/maruskyve
@ Description:
@ Note: -
@ Dependency: -
"""

from random import shuffle
from src.py.utils.array.array_operation import transpose


class __Splitter:
    """
    A class that contains method for data split
    """

    def __init__(self):
        pass

    def fit_array(self,
                  array,
                  row_a=None,
                  row_b=None,
                  col_a=None,
                  col_b=None):
        arr = array.copy()
        start_index = 0

        row_a = start_index if not row_a else row_a
        row_b = len(array) if not row_b else row_b  # Index for last row
        col_a = start_index if not col_a else col_a
        col_b = len(array[0]) if not col_b else col_b  # Index for last column

        arr = arr[row_a:row_b]
        arr = transpose(arr)[col_a:col_b]
        arr = transpose(arr)

        return arr

    def train_test_split(self,
                         x_array,  # Feature
                         y_array,  # Label
                         train_size=None,
                         test_size=None,
                         random_state: bool = None):

        if random_state:  # Shuffle x and y array for random_state
            shuffle(x_array)
            shuffle(y_array)

        # Split each x and y in context of split by !random_state
        index = round(len(x_array) * train_size) if train_size else len(x_array)-round(len(x_array) * test_size)
        x_train = x_array[0:index]
        x_test = x_array[index:len(x_array)]
        y_train = y_array[0:index]
        y_test = y_array[index:len(y_array)]

        return x_train, x_test, y_train, y_test


__splitter = __Splitter()
fit_array = __splitter.fit_array
train_test_split = __splitter.train_test_split

# Dummy data
test_arr = [['x', 'y', 'kategori'],
            [7.0, 6.0, 'bad'],
            [6.0, 6.0, 'bad'],
            [6.0, 5.0, 'bad'],
            [1.0, 3.0, 'bad'],
            [2.0, 4.0, 'good'],
            [2.0, 4.0, 'good'],
            [2.0, 2.0, 'good']]
# print(__Splitter().fit_array(array=test_arr))
# x = fit_array(array=test_arr, row_a=1, col_b=-1)
# y = fit_array(array=test_arr, row_a=1, col_a=-1)
# x_train, x_test, y_train, y_test = train_test_split(x_array=x, y_array=y, test_size=0.7)
# print(x_test)
