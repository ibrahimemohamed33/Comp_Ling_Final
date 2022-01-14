import inflect
import pickle
import os
import data
import eng_to_ipa as ipa
import pandas as pd

from constants import *


def word_IPA(word: str) -> str:
    '''
    Returns a word's IPA if it is within the CMU dictionary
    '''
    if ipa.isin_cmu(word):
        return ipa.convert(word).replace('ˈ', '').replace('ˌ', '')
    return EMPTY_CHARACTER


def contains_empty_characters(*args) -> bool:
    '''
    Checks for empty characters
    '''
    has_empty_char, index, arg_length = False, 0, len(args)
    while not has_empty_char and index < arg_length:
        has_empty_char = (args[index] == EMPTY_CHARACTER)
        index += 1
    return has_empty_char


def replace_suffix(word: str, char: str) -> str:
    '''
    Replaces the suffix of word with char
    '''
    return word[: len(word) - len(char)] + char


def format_noun(noun: str) -> str:
    '''
    Formats the regular noun based on its suffix
    '''
    if noun[-1] == 'y' and noun[-2] not in vowels:
        noun = replace_suffix(noun, 'i')
    return noun.lower()


def get_object_path(name: str, folder_name: str, is_phonology=False) -> str:
    '''
    Returns the directory of an object with the name depending on these params
    '''
    phon_name = data.PHONOLOGY_NAME if is_phonology else ''
    object_name = phon_name + '_' + name
    return os.path.join(folder_name, object_name)


def load_object(name: str, return_dict: bool,
                folder_name=PHONEME_FOLDER_NAME) -> dict or list:
    '''
    Loads an object using pickle given the object's name and folder
    '''
    object_path = get_object_path(name, folder_name)

    if not os.path.isfile(object_path):
        return {} if return_dict else []

    with open(object_path, 'rb') as f:
        return pickle.load(f)


def save_object(name: str, object: list or dict or pd.DataFrame, is_phon_model=None,
                folder_name=PHONEME_FOLDER_NAME) -> None:
    '''
    Saves an object into the folder with a given name
    '''
    path = get_object_path(name, folder_name, is_phon_model)
    with open(path, 'wb') as f:
        pickle.dump(object, f)


def process_nouns(nouns: 'list[str]', is_phon_model: bool,
                  engine: inflect.engine):
    '''
    Processes and saves the nouns into an object file. This significantly
    reduces the time for each model to analyze > 15,000 nouns, as well as allows
    for reproduction in results.
    '''
    if not is_phon_model:
        return nouns, {}

    phoneme_dict = load_object(PHONEME_DICT_NAME, True)
    # updates the phoneme_dict with the new nouns
    if len(phoneme_dict) != len(nouns):
        for noun in nouns:
            noun_rep, plural_rep = create_rep(noun, engine, is_phon_model)
            if noun_rep and plural_rep and noun_rep not in phoneme_dict:
                phoneme_dict[noun_rep] = plural_rep

        phoneme_list = list(phoneme_dict.keys())
        save_object(PHONEME_DICT_NAME, phoneme_dict)
        save_object(PHONEME_LIST_NAME, phoneme_list)

    return phoneme_list, phoneme_dict


def create_rep(noun: str, engine: inflect.engine, is_phon_model: bool):
    '''
    Creates the noun's orthographic or phonological representation.
    If neither representation is returned, then the function returns None to
    ensure parity between the models
    '''
    noun_sing = noun.replace('-', '').replace('_', '')
    noun_plural = engine.plural_noun(noun_sing)
    (noun_rep, plural_rep) = (word_IPA(noun_sing), word_IPA(noun_plural))

    if contains_empty_characters(noun_rep, plural_rep):
        return None, None

    return (noun_rep, plural_rep) if is_phon_model else (noun_sing, noun_plural)


def remove_unsimilar_words(df: pd.DataFrame):
    '''
    Removes words from the dataframe which lack any similar words
    '''
    return df[df.similar_words.map(lambda d: len(d) > 0)]


def is_singular_noun(word: str, pos: str):
    '''
    Checks if noun is singular
    '''
    return len(word) > 2 and pos == 'NN' and word != "sherlock"


def calculate_accuracy_rate(values: dict):
    try:
        return values['True'] / (values['True'] + values['False'])
    except KeyError as e:
        return 0 if e == "True" else 1
