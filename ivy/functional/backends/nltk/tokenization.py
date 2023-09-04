import nltk

def tokenize(text):
    """
    Tokenize a text using NLTK.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list of str: A list of tokens extracted from the input text.
    """
    tokens = nltk.word_tokenize(text)
    return tokens

def sentence_tokenize(text):
    """
    Tokenize a text into sentences using NLTK.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list of str: A list of sentences extracted from the input text.
    """
    sentences = nltk.sent_tokenize(text)
    return sentences

