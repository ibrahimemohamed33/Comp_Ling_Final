import inflect
import pickle
import os
import data
import eng_to_ipa as ipa
import pandas as pd


vowels = ['a', 'e', 'i', 'o', 'u']

EMPTY_CHARACTER, DELETE = "*", '-'
PHONEME_DICT_NAME = "phoneme_dict.obj"
PHONEME_LIST_NAME = 'phoneme_list.obj'
PHONEME_FOLDER_NAME = 'Object_Files'
MODEL_COLUMNS = ["noun", 'noun_rep', 'similar_words', "predicted_plural",
                 'actual_plural', 'accurate_prediction']


def word_IPA(word: str) -> str:
    '''
    Checks if a given word is in the CMU dictionary and then returns its
    corresponding IPA
    '''
    return ipa.convert(word) if ipa.isin_cmu(word) else EMPTY_CHARACTER


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


def get_object_path(name: str, is_regular: bool, folder_name: str,
                    is_phonology=False) -> str:
    '''
    Returns the directory of an object with the name depending on these params
    '''
    regular_name = data.REGULAR_NAME if is_regular else data.IRREGULAR_NAME
    phon_name = data.PHONOLOGY_NAME if is_phonology else ''
    object_name = regular_name + '_' + phon_name + '_' + name
    return os.path.join(folder_name, object_name)


def load_object(name: str, is_regular: bool, is_dict: bool,
                folder_name=PHONEME_FOLDER_NAME) -> dict or list:
    '''
    Loads an object using pickle given the object's name and folder
    '''
    path = get_object_path(name, is_regular, folder_name)
    if not os.path.isfile(path):
        return {} if is_dict else []

    with open(path, 'rb') as f:
        return pickle.load(f)


def save_object(name: str, is_regular: bool, object: list or dict, is_phonology=None,
                folder_name=PHONEME_FOLDER_NAME) -> None:
    '''
    Saves an object into the folder with a given name
    '''
    path = get_object_path(name, is_regular, folder_name, is_phonology)

    with open(path, 'wb') as f:
        pickle.dump(object, f)


def process_nouns(nouns: 'list[str]', is_phonology: bool, engine: inflect.engine,
                  is_regular: bool):
    '''
    Processes and saves the nouns into an object file. One huge advantage was
    this reduces the HOURS of computing that goes into these large datasets.
    Another one is that it allows anyone with the file to load up the data and
    reproduce the results
    '''
    if not is_phonology:
        return nouns, {}
    else:
        phoneme_dict = load_object(PHONEME_DICT_NAME, is_regular, True)
        if len(phoneme_dict) != len(nouns):
            for noun in nouns:
                noun_rep, plural_rep = create_rep(noun, engine, is_phonology)
                if noun_rep and plural_rep and noun_rep not in phoneme_dict:
                    phoneme_dict[noun_rep] = plural_rep

            phoneme_list = list(phoneme_dict.keys())
            save_object(PHONEME_DICT_NAME, is_regular, phoneme_dict)
            save_object(PHONEME_LIST_NAME, is_regular, phoneme_list)

        return phoneme_list, phoneme_dict


def create_rep(noun: str, engine: inflect.engine, is_phonology: str):
    '''
    Creates an orthographic or phonetic representation for each noun. If one
    or the other does not hold, we return None to ensure they do not get loaded
    into the models
    '''
    noun = noun.replace('-', '').replace('_', '')
    plural_noun = engine.plural_noun(noun)
    noun_rep = word_IPA(noun).replace('ˈ', '').replace('ˌ', '')
    plural_rep = word_IPA(plural_noun).replace('ˈ', '').replace('ˌ', '')

    if noun_rep == EMPTY_CHARACTER or plural_rep == EMPTY_CHARACTER:
        return None, None

    return (noun_rep, plural_rep) if is_phonology else (noun, plural_noun)


def remove_unsimilar_words(df: pd.DataFrame):
    '''
    Removes words from the dataframe which lack any similar words
    '''
    return df[df.similar_words.map(lambda d: len(d) > 0)]
