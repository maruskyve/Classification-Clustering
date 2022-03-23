"""
Created on Thu March 03 17:09:39 2022
@ Author: Ma'ruf Nurrochman/maruskyve
@ Description: 
    Reusable K-Nearest Neighbors (KNN) model for predicting a neighbors data 
    with given test data using the value of K and the distance metrics from user
@ Note:
    Recommended to use odd number for K-value
@ Dependency: Numpy
"""

import numpy as np
import time
import src.py.utils.utils as utl


class KNearestNeighbors:
    def __init__(self,
                 k_val: int,
                 distance_metrics: str):
        self.data = df  # User given information
        self.model_k = k_val  # User given information
        self.model_distance_metrics = distance_metrics  # User given information

        self.x_train = list()  # User given information
        self.x_test = list()  # User given information
        self.y_train = list()  # User given information
        self.y_test = list()  # User given information

        self.model_class = list()
        self.model_distances = list()
        self.model_nearest_distances = list()
        self.model_nearest_neighbors = list()
        self.model_class_poll = list()
        self.model_y_textual_prediction = list()
        self.model_y_prediction_encoded = list()
        self.model_accuracy = float()

        self.learning_time = float()

    def __notifier(self):
        if self.model_k > len(self.y_test):
            print('The process aborted because the K-value is more than the given test data (y_test)')

    def __data_preps(self, x_train, x_test, y_train, y_test):
        """
        Method (private) to prepare given learning data

        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :return:
        """

        print(' - Preparing learning data')

        # On user create model
        self.x_train = x_train
        self.x_test = x_test

        # Model Y Train
        self.y_train = utl.flatten(array=y_train)
        self.y_train = utl.text_noise_chars_remover(text=self.y_train)

        # Model Y Test
        self.y_test = utl.flatten(array=y_test)
        self.y_test = utl.text_noise_chars_remover(text=self.y_test)

        # On runtime
        # Model klass
        self.model_class = utl.combine(y_train, y_test)
        self.model_class = utl.flatten(array=self.model_class)
        self.model_class = utl.text_noise_chars_remover(text=self.model_class)
        self.model_class = utl.class_extractor(class_array=self.model_class)

    def __compute_distance(self,
                           x_train_vector,
                           x_test_vector):
        """
        Method (private) to compute the distance between two data points
        based on the feature value.

        Only accept 1-Dimension for both of x_train and x_test (vector)
        :param x_train_vector:
        :param x_test_vector:
        :return:
        """

        print(' - Compute distance')

        x_train = x_train_vector
        x_test = x_test_vector
        distances = utl.zeros(len(x_train))

        if self.model_distance_metrics == 'euclidean':
            for i in range(len(x_train)):  # Context on Vector / row of x_train array
                for j in range(len(x_train[i])):  # Context on Scalar / column x_train array
                    distances[i] += (float(x_test[j]) - float(x_train[i][j])) ** 2
                distances[i] = utl.math.sqrt(distances[i])

        elif self.model_distance_metrics == 'manhattan':
            for i in range(len(x_train)):
                for j in range(len(x_train[i])):
                    distances[i] += abs(float(x_test[j]) - float(x_train[i][j])) ** 2
                distances[i] = utl.math.sqrt(distances[i])

        elif self.model_distance_metrics == 'chebyshev':
            for i in range(len(x_train)):
                feature_distances = list()

                for j in range(len(x_train[i])):  # Context on Scalar / column x_train array
                    feature_distances.append(abs(float(x_test[j]) - float(x_train[i][j])))
                distances[i] = max(feature_distances)

        return distances

    def __find_nearest_neighbors(self,
                                 distance_vector: list,
                                 y_train_vector: list,
                                 k_val: int,
                                 priority: int = None):
        """
        Method (private) to find nearest neighbors

        :param distance_vector:
        :param y_train_vector:
        :param k_val:
        :param priority:
        :return:
        """

        print(' - Search for nearest neighbors')

        priority = 0 if not priority else priority
        distances_vector_cpy = distance_vector.copy()
        y_train = y_train_vector
        nearest_distances = utl.zeros(n=k_val)
        nearest_neighbors = utl.zeros(n=k_val)

        for i in range(k_val):  # Searching for nearest neighbors start
            min_index = distances_vector_cpy.index(min(distances_vector_cpy))
            nearest_distances[i] = min(distances_vector_cpy)
            nearest_neighbors[i] = y_train[min_index]
            distances_vector_cpy[min_index] = max(distances_vector_cpy) + min(distances_vector_cpy)

        return nearest_distances, nearest_neighbors

    def __class_poll(self,
                     class_vector: list,
                     nearest_neighbors: list):
        """
        Method (private) to polling for the winner class based on the nearest neighbors

        :param class_vector:
        :param nearest_neighbors:
        :return:
        """

        print(" - Polling class")

        class_poll = utl.zeros(len(class_vector))

        for x in nearest_neighbors:
            class_index = class_vector.index(x)
            class_poll[class_index] += 1

        return class_poll

    def __y_textual_predict(self,
                            class_vector: list,
                            class_poll: int):
        """
        Method (private) to predict data class using argmax based on the class poll value
        Note: currently depend on numpy.argmax()

        :param class_vector:
        :param class_poll:
        :return:
        """

        print(" - Predict result (textual)")

        predict = class_vector[np.argmax(class_poll)]

        return predict

    def __encode_y_textual_predict(self,
                                   class_vector,
                                   y_textual_predict):
        """
        Method (private)to encode predict result to binary mode / one hot encoder

        :param class_vector:
        :param y_textual_predict:
        :return:
        """

        print(" - Encoding textual predicted result")

        class_encoded = utl.one_hot_encoder(class_vector=class_vector,
                                            actual_predict_vector=y_textual_predict)

        return class_encoded

    def __model_accuracy(self,
                         class_vector,
                         y_test_vector,
                         y_predict_encoded):
        """
        Method (private) to compute model accuracy

        :param class_vector:
        :param y_test_vector:
        :param y_predict_encoded:
        :return:
        """

        print(" - Compute model accuracy: ")

        y_test = y_test_vector
        accuracy_vector = list()

        for i in range(len(y_test)):  # Accessing y_test, equal to length of y_predict_encoded
            class_index = class_vector.index(y_test[i])  # Index of class array in y_test[i]
            accuracy_vector.append(1 if y_predict_encoded[i][class_index] == 1 else 0)

            # print(y_test[i], class_index, y_predict_encoded[i][class_index])
        accuracy = sum(accuracy_vector) / len(accuracy_vector)

        return accuracy

    def quick_build(self,
                    data_frame: list,
                    train_test_random_state: bool = None):
        """
        :param data_frame:
        :param train_test_random_state:
        :return:
        """
        pass

    # Non reusable
    def build(self,
              x_train: list,
              x_test: list,
              y_train: list,
              y_test: list):  # Fit and Predict
        """
        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :return:
        """
        start = time.time()

        self.__data_preps(x_train=x_train,
                          x_test=x_test,
                          y_train=y_train,
                          y_test=y_test)

        print("Processing learning")
        for i in range(len(self.x_test)):  # Compute distance for each x_test
            print(f" Process [{i + 1}]")
            self.model_distances.append(self.__compute_distance(x_train_vector=self.x_train,
                                                                x_test_vector=self.x_test[i]))
            find_nearest_neighbors = self.__find_nearest_neighbors(distance_vector=self.model_distances[i],
                                                                   y_train_vector=self.y_train,
                                                                   k_val=self.model_k,
                                                                   priority=0)
            self.model_nearest_distances.append(find_nearest_neighbors[0])
            self.model_nearest_neighbors.append(find_nearest_neighbors[1])
            self.model_class_poll.append(self.__class_poll(class_vector=self.model_class,
                                                           nearest_neighbors=self.model_nearest_neighbors[i]))
            self.model_y_textual_prediction.append(self.__y_textual_predict(class_vector=self.model_class,
                                                                            class_poll=self.model_class_poll[i]))
        print(' Finalization')
        self.model_y_prediction_encoded = self.__encode_y_textual_predict(class_vector=self.model_class,
                                                                          y_textual_predict=self.model_y_textual_prediction)
        self.model_accuracy = self.__model_accuracy(class_vector=self.model_class,
                                                    y_test_vector=self.y_test,
                                                    y_predict_encoded=self.model_y_prediction_encoded)
        self.learning_time = time.time() - start

    def __summary(self, exec_time: float):
        """
        :param exec_time:
        :return:
        !!! TEMPORARY PLANNED TO DELETE !!!
        Summarize given information and result
        """
        print(' - Summarizing result')

        class_poll = np.zeros(len(self.model_class))

        for x in self.model_nearest_neighbors:  # Class polling
            class_index = np.where(self.model_class == x)
            class_poll[class_index] += 1

        # Predicting class by class poll argmax/index from max value
        self.prediction = self.model_class[np.argmax(class_poll)]
        summary = f"\n### Summary ###" \
                  f"\n> Data total: {len(self.model_nearest_neighbors)}" \
                  f"\n> Detected class, (total: {len(self.model_class)}):"

        for x in self.model_class:
            summary += f'\n   - {x}'

        summary += f"\n> Test data, (total: {len(self.y_test)}): "

        for i in range(len(self.y_test)):
            summary += f'\n   - x{i + 1} = {self.y_test[i]}'

        summary += f"\n> K-value: {self.model_k}" \
                   f"\n> Distance metrics: {self.model_distance_metrics}" \
                   f"\n> Class Poll:"

        for i in range(len(class_poll)):
            summary += f"\n   - {self.model_class[i]} = {class_poll[i]}"
        summary += f"\n> Time spent: {round(exec_time, 7)} seconds" \
                   f"\n> Prediction: {self.prediction}" \
                   f"\n### Summary ###"
        print(summary)


# Data frame and both (train & test) data initialization
df = utl.csv_reader(filepath='../../../assets/dataset/iris.csv')
df = utl.transpose(utl.transpose(df)[1: len(df)])  # 1st column reduction
_x = utl.fit_array(array=df, row_a=1, col_a=1, col_b=2)
_y = utl.fit_array(array=df, row_a=1, col_a=-1)

# H-param
k = 100
dist_metrics = "euclidean"

xtr, xte, ytr, yte = utl.train_test_split(_x, _y, train_size=0.7, random_state=True)
model = KNearestNeighbors(k_val=k,
                          distance_metrics=dist_metrics)
model.build(x_train=xtr,
            x_test=xte,
            y_train=ytr,
            y_test=yte)

print('Class: ', model.model_class)
print('YTest: ', model.y_test)
print('Encoded', model.model_y_prediction_encoded)
print('Accuracy: ', model.model_accuracy)
print('Learning time: ', model.learning_time)
