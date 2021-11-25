import nltk
import inflect

from collections import Counter
from string import ascii_lowercase, punctuation

REGULAR_NAME = 'regular'
IRREGULAR_NAME = 'irregular'
PHONOLOGY_DATA_NAME = "phonologies"
ORTHOGRAPH_DATA_NAME = 'orthography'

vowels = ['a', 'e', 'i', 'o', 'u']
letter_to_number = {letter: index for index,
                    letter in enumerate(list(ascii_lowercase))}

number_to_letter = {index: letter for index,
                    letter in letter_to_number.items()}

phoneme_to_number = {}

number_to_phoneme = {}


def replace_suffix(word: str, char: str) -> str:
    return word[: len(word) - len(char)] + char


def get_name(is_regular: bool, is_phonology: bool):
    nameA = '_' + (REGULAR_NAME if is_regular else IRREGULAR_NAME) + '.txt'
    return (PHONOLOGY_DATA_NAME if is_phonology else ORTHOGRAPH_DATA_NAME) + nameA


def format_noun(noun: str) -> str:
    if noun[-1] == 'y' and noun[-2] not in vowels:
        noun = replace_suffix(noun, 'i')
    return noun.lower()


def is_regular_noun(inflect_engine: inflect.engine, noun: str) -> bool:
    '''
    Checks if a noun is regular by seeing if it's plural form ends with the
    suffix '-s' or '-es,' following standard definitions. This can be checked
    by checking first if the suffix ends with an '-es' or '-s', or if it behaves
    one of the more obscure rules when ending with an '-y', '-f', or '-fe'

    '''
    formatted_noun = format_noun(noun)
    pluralized_noun = inflect_engine.plural_noun(noun).lower()
    suffixA, suffixB = 'es', 's'

    return (
        (formatted_noun + suffixA == pluralized_noun) or
        (formatted_noun + suffixB == pluralized_noun)
    )


def find_all_singular_nouns(filename: str) -> 'list[str]':
    plural_engine = inflect.engine()
    text = open(filename).read()
    tokens = nltk.word_tokenize(text.lower())
    # returns all singular regular and irregular nouns in the text corpus
    regular_nouns, irregular_nouns = [], []
    for (word, pos) in nltk.pos_tag(tokens):
        if len(word) > 2 and pos == 'NN' and word != "sherlock":
            if is_regular_noun(plural_engine, word):
                regular_nouns.append(word)
            else:
                irregular_nouns.append(word)
    return Counter(regular_nouns).keys(), Counter(irregular_nouns).keys()


def convert_string_to_index(word: str, dictionary: dict):
    formatted_word = word.translate(str.maketrans('', '', punctuation))
    return [dictionary[char] for char in list(formatted_word)]
