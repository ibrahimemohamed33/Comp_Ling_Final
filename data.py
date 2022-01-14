import nltk
import inflect
import word_utils
import os
import re

from collections import Counter
from constants import *


def is_regular_noun(noun: str, engine: inflect.engine) -> bool:
    '''
    Checks if a noun's plural suffix is '-s' or '-es,' following standard definitions.
    If so, then the noun is deemed regular 
    '''
    formatted_noun = word_utils.format_noun(noun)
    pluralized_noun = engine.plural_noun(noun).lower()
    suffixA, suffixB = REGULAR_SUFFIXES

    return (
        (formatted_noun + suffixA == pluralized_noun) or
        (formatted_noun + suffixB == pluralized_noun)
    )


def find_all_singular_nouns(filename: str, engine: inflect.engine) -> 'tuple[list, list]':
    '''
    Finds all unique, singular nouns in the file using NLTK's pos_tag feature. The
    function then classifies each noun as regular or irregular, and orders
    nouns by their frequency
    '''
    file_path = os.path.join(TEXT_FOLDER_NAME, filename)
    text_content = open(file_path).read().lower()
    tokens = nltk.word_tokenize(text_content)

    # returns all singular regular and irregular nouns in the text corpus
    reg_nouns, irreg_nouns = [], []
    for (word, pos) in nltk.pos_tag(tokens):
        formatted_word = re.sub(r'\W+', '', word.replace('-', ''))
        is_singular = word_utils.is_singular_noun(formatted_word, pos)
        if is_singular:
            if is_regular_noun(formatted_word, engine):
                reg_nouns.append(formatted_word)
            else:
                irreg_nouns.append(formatted_word)

    unique_reg, unique_irreg = Counter(reg_nouns), Counter(irreg_nouns)
    unique_reg = list(unique_reg.keys())
    unique_irreg = list(unique_irreg.keys())

    return unique_reg, unique_irreg
