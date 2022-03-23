"""
Created on Thu March 03 17:09:39 2022
@ Author: Ma'ruf Nurrochman/maruskyve
@ Description:
@ Note: -
@ Dependency: -
"""


class __Cleaner:
    noise_chars = ['\n', '\t', '[', ']']

    def __init__(self):
        pass

    def text_noise_chars_remover(self,
                                 text,
                                 noise_chars=None):
        noise_chars = self.noise_chars if not noise_chars else noise_chars
        output = str() if type(text).__name__ == 'str' else list()

        def a_text(txt):
            text_cpy = txt
            for x in noise_chars:  # Removing possible noisy characters
                if x in text_cpy:
                    text_cpy = text_cpy.replace(x, '')
                else:
                    pass

            return text_cpy

        if type(text).__name__ == 'str':  # Detect if type of text is str
            output = a_text(text)

        elif type(text).__name__ == 'list':  # Detect if type of text is list
            for x in text:
                output.append(a_text(x))

        return output


# test_text_array = ['text1\n', '[text2]', 'text3\t']
# test_text = '[tesxt\n]'

__cleaner = __Cleaner()
text_noise_chars_remover = __cleaner.text_noise_chars_remover

# print(text_noise_chars_remover(test_text_array))
