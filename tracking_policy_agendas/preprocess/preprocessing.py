"""
Preprocessing

....................................................................................................
MIT License
Copyright (c) 2021-2023 AUT Iran, Mohammad H Forouhesh
Copyright (c) 2021-2022 MetoData.ai, Mohammad H Forouhesh
....................................................................................................
This module contains various tools for text preprocessing.
"""

import re


def remove_emoji(text: str) -> str:
    """
    A function to remove emojis using regex
    :param text:    An input text.
    :return:        A text with removed emojis
    """
    emoji_pattern = re.compile(pattern="["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       u"\U00002702-\U000027B0"
                                       u"\U000024C2-\U0001F251"
                                       u"\U0001F300-\U0001F5FF"
                                       u"\U0001F1E6-\U0001F1FF"
                                       u"\U00002700-\U000027BF"
                                       u"\U0001F900-\U0001F9FF"
                                       u"\U0001F600-\U0001F64F"
                                       u"\U0001F680-\U0001F6FF"
                                       u"\U00002600-\U000026FF"
                                       u'\u200d'
                                       u'’'
                                       u'£'
                                       u'\u2060-\u2069'
                                       u'í'
                                       u'ó'
                                       u'ú'
                                       u'á'
                                       u'–'
                                       u'“”‘‘‘'
                                       u'éàééàéééàéè'
                                       u'üöççəəəəçä'
                                       u'ışşƏıışşƏışêêñ'
                                       u'İğğ~•'
                                       u'⏯'
                                       "]+", flags=re.UNICODE)
    try:    return str(emoji_pattern.sub(r'', text))
    except: return ''


def remove_redundant_characters(text: str) -> str:
    """
    A tool to remove redundant and unwanted characters
    :param text:    An input text.
    :return:        A text with removed unwanted characters (punctuations, latin, etc.)
    """
    text = text.replace('\u200c', ' ')
    text = re.sub(r'@[A-Za-z0-9]+', ' ', text)  # Removed @mentions
    text = re.sub(r'_[A-Za-z0-9]+', ' ', text)  # Removed underlines
    text = re.sub(r'/(\r\n)+|\r+|\n+|\t+/', ' ', text)  # Removed \n
    text = re.sub(r'#', ' ', text)  # Removing the '#' symbol
    text = re.sub(r'RT[\s]+', ' ', text)  # Removing RT
    text = re.sub(r'https?:\/\/\S+', ' ', text)  # Remove the hyper link
    text = re.sub(r'\([ا-ی]{1,3}\)', ' ', text)  # Remove abbreviations
    text = re.sub(r"[\(\)]", " ", text)  # remove parantesis
    text = re.sub(r"\d|[۰-۹]", " ", text)
    text = re.sub(r"&|:", " ", text)
    # text = re.sub(r"[A-Za-z]", " ", text)
    # text = re.sub(r"[0-9]", " ", text)
    text = re.sub(r"\"", " ", text)
    text = re.sub(r"\'", " ", text)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"—", " ", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"@|=", " ", text)
    text = re.sub(r"^\d+\s|\s\d+\s|\s\d+$", " ", text)
    text = re.sub(r"{|}|;|\[|\]|\||؟|!|\+|\-|\*|\$", " ", text)
    text = re.sub(r"¹²|\/", " ", text)
    text = re.sub(r"»|>|<|«|,|؛|،|%|؟", " ", text)
    text = re.sub("\.|\^|,", " ", text)
    text = text.replace('…', ' ')
    text = text.replace('?', ' ')
    # text = ' '.join(list(map(lambda word: '' if len(word) < 3 else word, text.split())))
    # return ' '.join([word for word in text.split(' ') if len(word) > 1
    #                  and False not in [char in 'آ ا ب پ ت ث ج چ ح خ د ذ ر ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ه ی'
    #                                    for char in word]
    #                  and len(text.split(' ')) > 1])
    return text
