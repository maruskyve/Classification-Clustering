# A class contains method to transpose 2D array
from src.py.utils.array.extractor import class_extractor


class __Operation:
    def __init__(self):
        pass

    def transpose(self, array):
        return [[array[j][i] for j in range(len(array))]
                for i in range(len(array[0]))]

    def zeros(self,
              n=None,
              shape=None):
        arr = list()

        if shape:
            for d1 in range(shape[0]):
                temp_arr = list()
                for d2 in range(shape[1]):
                    temp_arr.append(.0)
                arr.append(temp_arr)

        else:
            arr = [float(0) for _ in range(n)]
        return arr

    def ones(self,
             n=None,
             shape=None):
        arr = list()

        if shape:
            for d1 in range(shape[0]):
                temp_arr = list()
                for d2 in range(shape[1]):
                    temp_arr.append(.0)
                arr.append(temp_arr)

        else:
            arr = [float(1) for _ in range(n)]
        return arr

    def flatten(self,
                array):
        array_cpy = array.copy()
        array_cpy = [item for sublist in array_cpy for item in sublist]

        return array_cpy

    def combine(self,
                array_1,
                array_2):
        array_1_cpy = array_1.copy()
        array_2_cpy = array_2.copy()
        output_array = list()

        for x in array_1_cpy:
            output_array.append(x)

        for x in array_2_cpy:
            output_array.append(x)

        return output_array

    def one_hot_encoder(self,
                        class_vector: list,
                        actual_predict_vector: list):
        # Method to encode predict result to binary mode / one hot encoder
        # With y_predict as row, class_array as column
        # Note: Require 1D class_array and y_class_predict
        """
        Method to encode predict result to binary mode / one hot encoder,
        with y_predict as row and class array as column,
        Note: this method require 1-Dimension of both class_vector and actual_predict_vector

        :param class_vector:
        :param actual_predict_vector:
        :return:
        """

        class_encoded = self.zeros(shape=(len(actual_predict_vector),
                                          len(class_vector)))

        for i in range(len(actual_predict_vector)):
            for j in range(len(class_vector)):
                index = class_vector.index(actual_predict_vector[i])
                class_encoded[i][index] = 1
                class_encoded[i][j] = int(class_encoded[i][j])  # To Integer

        return class_encoded

    def confusion_matrix(self,
                         actual_predict_vector: list,
                         encoded_predict_vector: list):
        """
        :param actual_predict_vector:
        :param encoded_predict_vector:
        :return:
        !!! NOT FINISHED YET !!!
        Example of expected structure for encoded array
        label = ['bad', 'good']
        label_size = 2
        encoded_array =
            [[0, 1]
             [1, 0],
             [1, 0],
             [1, 0]]

        Example of expected structure for c_matrix
        [[1, 1],
         [0, 1]]
        """
        class_array = class_extractor(actual_predict_vector)
        encoded_actual = self.one_hot_encoder(class_array, actual_predict_vector)
        class_len = len(class_array)
        encoded_len = len(encoded_predict_vector)
        c_matrix = self.zeros(shape=(class_len, class_len))
        print('class: ', class_array)
        print('encoded actual: ', encoded_actual)
        print('encoded predict: ', encoded_predict_vector, '\n')

        for i in range(class_len):
            for j in range(class_len):
                for i1 in range(encoded_len):
                    print(encoded_actual[i1], encoded_predict_vector[i1])
                    c_matrix[i][j] += 1 if encoded_predict_vector[i1] == encoded_actual[i1] else 0
                    # c_matrix[i][j] += 1 if encoded_predict_array[i1][j] == 1 else 0

        print('\n', c_matrix)

        return 0


string_arr = [['bad\n'], ['good\n'], ['good']]

string_arr_2 = [['good\n'], ['good']]

__operation = __Operation()
transpose = __operation.transpose
zeros = __operation.zeros
ones = __operation.ones
flatten = __operation.flatten
combine = __operation.combine
confusion_matrix = __operation.confusion_matrix
one_hot_encoder = __operation.one_hot_encoder


# actual = ['bad', 'good', 'bad', 'good']
# encoded = [[0, 1],
#            [1, 0],
#            [1, 0],
#            [1, 0]]
# confusion_matrix(actual_predict_vector=actual,
#                  encoded_predict_vector=encoded)
