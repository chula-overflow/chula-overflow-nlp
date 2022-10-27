import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def sentence_tokenize(paragraph):
    """
    Input - (str) paragraph of question
    Output - (list of str) each sentence in list
    """
    paragraph = paragraph.replace(",", ".")
    sentences = sent_tokenize(paragraph)   
    sentences.reverse()
    return sentences


print(sentence_tokenize("A sphere is measured to have a radius of 10m with an error of +- 0.5m, find the margin error of the volume of that sphere."))