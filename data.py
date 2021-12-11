import nltk
import inflect
import word_utils
import os

from re import sub
from collections import Counter


TEXT_FOLDER_NAME = 'Text_Files'
REGULAR_NAME = 'regular'
IRREGULAR_NAME = 'irregular'
PHONOLOGY_NAME, ORTHOGRAPH_NAME = "phonology", 'orthography'


def get_file_pathname(is_regular: bool, is_phonology: bool, file_name: str) -> str:
    '''
    Returns the path name of the file depending if it is regular/irregular and
    phonological/orthographic
    '''
    noun_name = REGULAR_NAME if is_regular else IRREGULAR_NAME
    phonology_name = PHONOLOGY_NAME if is_phonology else ORTHOGRAPH_NAME
    nameA = '_' + noun_name + '_' + file_name
    return phonology_name + nameA


def is_regular_noun(noun: str, engine: inflect.engine) -> bool:
    '''
    Checks if a noun is regular by seeing if it's plural form ends with the
    suffix '-s' or '-es,' following standard definitions. This can be checked
    by checking first if the suffix ends with an '-es' or '-s', or if it behaves
    one of the more obscure rules when ending with an 'y'

    '''
    formatted_noun = word_utils.format_noun(noun)
    pluralized_noun = engine.plural_noun(noun).lower()
    es_suffix, s_suffix = 'es', 's'
    return (
        (formatted_noun + es_suffix == pluralized_noun) or
        (formatted_noun + s_suffix == pluralized_noun)
    )


def find_all_singular_nouns(filename: str) -> 'tuple[list, list]':
    '''
    Finds all singular nouns in the file by leveraging NLTK's pos_tag feature
    and then classifying the noun as regular or irregular. The function orders
    nouns by frequency using the Counter function and then returns the unique
    words
    '''
    engine = inflect.engine()
    text = open(os.path.join(TEXT_FOLDER_NAME, filename)).read()
    tokens = nltk.word_tokenize(text.lower())
    # returns all singular regular and irregular nouns in the text corpus
    reg_nouns, irreg_nouns = [], []
    for (word, pos) in nltk.pos_tag(tokens):
        word = sub(r'\W+', '', word.replace('-', ''))
        # strange bug where 'sherlock' prevents the data from being processed
        if len(word) > 2 and pos == 'NN' and word != "sherlock":
            array = reg_nouns if is_regular_noun(word, engine) else irreg_nouns
            array.append(word)

    unique_reg, unique_irreg = Counter(reg_nouns), Counter(irreg_nouns)
    return list(unique_reg.keys()), list(unique_irreg.keys())
