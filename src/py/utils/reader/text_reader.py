"""
Created on Thu March 03 17:09:39 2022
@ Author: Ma'ruf Nurrochman/maruskyve
@ Description:
@ Note: -
@ Dependency: -
"""

from src.py.utils.strings.cleaner import text_noise_chars_remover


class __TextReader:
    """
    Class for read data from text based file
    """

    def __init__(self):
        pass

    def csv_reader(self,
                   filepath: str):
        """
        Method (private) to read csv file data

        :param filepath:
        :return:
        """

        delimiter = ","  # Temporary until final result
        file_data = open(filepath, encoding='utf8')
        datas = list()

        for row in file_data:
            row_data = row.split(delimiter)

            for i in range(len(row_data)):  # Numeric data converting
                try:
                    row_data[i] = row_data[i]
                except Exception as e:
                    row_data[i] = text_noise_chars_remover(row_data[i])
            datas.append(row_data)
        datas = datas
        return datas


__text_reader = __TextReader()
csv_reader = __text_reader.csv_reader
# datax = __TextReader().csv_reader(filepath='../../../assets/dataset/iris.csv')
# print(datax)
