import nltk
import inflect
import re

from eng_to_ipa import isin_cmu
from collections import Counter
from string import ascii_lowercase, punctuation

REGULAR_NAME = 'regular'
IRREGULAR_NAME = 'irregular'
PHONOLOGY_NAME = "phonologies"
ORTHOGRAPH_NAME = 'orthography'

vowels = ['a', 'e', 'i', 'o', 'u']


def replace_suffix(word: str, char: str) -> str:
    return word[: len(word) - len(char)] + char


def get_name(is_regular: bool, is_phonology: bool, file_name: str):
    noun_name = REGULAR_NAME if is_regular else IRREGULAR_NAME
    phonology_name = PHONOLOGY_NAME if is_phonology else ORTHOGRAPH_NAME
    nameA = '_' + noun_name + '_' + file_name
    return phonology_name + nameA


def format_noun(noun: str) -> str:
    if noun[-1] == 'y' and noun[-2] not in vowels:
        noun = replace_suffix(noun, 'i')
    return noun.lower()


def is_regular_noun(inflect_engine: inflect.engine, noun: str) -> bool:
    '''
    Checks if a noun is regular by seeing if it's plural form ends with the
    suffix '-s' or '-es,' following standard definitions. This can be checked
    by checking first if the suffix ends with an '-es' or '-s', or if it behaves
    one of the more obscure rules when ending with an 'y'

    '''
    formatted_noun = format_noun(noun)
    pluralized_noun = inflect_engine.plural_noun(noun).lower()
    suffixA, suffixB = 'es', 's'

    return (
        (formatted_noun + suffixA == pluralized_noun) or
        (formatted_noun + suffixB == pluralized_noun)
    )


def find_all_singular_nouns(filename: str) -> 'list[str]':
    '''
    Finds all singular nouns in the file by leveraging NLTK's pos_tag feature
    and then classifying the noun as regular or irregular.
    '''
    plural_engine = inflect.engine()
    text = open(filename).read()
    tokens = nltk.word_tokenize(text.lower())
    # returns all singular regular and irregular nouns in the text corpus
    reg_nouns, irreg_nouns = [], []
    for (word, pos) in nltk.pos_tag(tokens):
        word = re.sub(r'\W+', '', word.replace('-', ''))
        if len(word) > 2 and pos == 'NN' and word != "sherlock":
            if is_regular_noun(plural_engine, word):
                reg_nouns.append(word)
            else:
                irreg_nouns.append(word)

    unique_reg, unique_irreg = Counter(reg_nouns), Counter(irreg_nouns)
    return list(unique_reg.keys()), list(unique_irreg.keys())


def convert_string_to_index(word: str, dictionary: dict):
    '''
    Formats string 
    '''
    formatted_word = word.translate(str.maketrans('', '', punctuation))
    return [dictionary[char.lower()] for char in list(formatted_word)]
