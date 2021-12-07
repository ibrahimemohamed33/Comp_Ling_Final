import inflect
import pickle
import eng_to_ipa as ipa
import os


vowels = ['a', 'e', 'i', 'o', 'u']

EMPTY_CHARACTER, DELETE = "*", '-'
PHONEME_DICT_NAME = "phoneme_to_phoneme_plural.obj"
PHONEME_DICT_FOLDER_NAME = 'Object_Files'
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


def process_nouns(nouns: 'list[str]', is_phonology: bool, engine: inflect.engine,
                  is_regular: bool):
    if not is_phonology:
        return nouns, {}
    else:
        regular_name = "REGULAR" if is_regular else "IRREGULAR"
        file_name = regular_name + "_" + PHONEME_DICT_NAME
        path = os.path.join(PHONEME_DICT_FOLDER_NAME, file_name)
        phoneme_dict, phoneme_list = {}, []
        if os.path.isfile(path):
            with open(file_name, 'rb') as f:
                phoneme_dict = pickle.load(f)

        for noun in nouns:
            plural_noun = engine.plural_noun(noun)
            noun_IPA, plural_IPA = word_IPA(noun), word_IPA(plural_noun)
            if noun_IPA != EMPTY_CHARACTER and plural_IPA != EMPTY_CHARACTER:
                phoneme_list.append(noun_IPA)
                if noun_IPA not in phoneme_dict:
                    phoneme_dict[noun_IPA] = plural_IPA

        with open(path, 'wb') as f:
            pickle.dump(phoneme_dict, f)

        return phoneme_list, phoneme_dict


def create_rep(noun: str, engine: inflect.engine, is_phonology: str):
    plural_noun = engine.plural_noun(noun)
    if not is_phonology:
        return noun, plural_noun
    return word_IPA(noun), word_IPA(plural_noun)
