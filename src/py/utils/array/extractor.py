class __Extractor:
    def __init__(self):
        pass

    def class_extractor(self,
                        class_array: list):
        detected_class = list()

        for x in class_array:  # Iterating target/class label set
            if x not in detected_class:
                detected_class.append(x)

        return detected_class


# arr = ['bad', 'good', 'good', 'dew', 'bad', 'bad', 'good']

__extractor = __Extractor()
class_extractor = __extractor.class_extractor
# print(class_extractor(arr))
