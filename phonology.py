import inflect
import math
import pandas as pd
import eng_to_ipa as ipa
import torch

from string import punctuation
from data import convert_string_to_index, get_name, letter_to_number

PHONE_LENGTH = 3
EMPTY_CHARACTER = "*"


def word_IPA(word: str) -> str:
    '''
    Checks if a given word is in the CMU dictionary and then returns its 
    corresponding IPA
    '''
    return ipa.convert(word) if ipa.isin_cmu(word) else ''


def phonetic_representation(noun: str) -> str:
    '''
    Represents a string following a similar approach to the Wickelfeature. This
    method essentially divides the string groups of PHONE_LENGTH syllables, and if the
    word is not divisble by PHONE_LENGTH, it carries the needed phonemes over to the
    remaining group. If the noun is smaller than PHONE_LENGTH, then it adds
    the empty character to signify that it should be ignored.

    EX (when PHONE_LENGTH = 3):
        i) Goose -> Goo | se -> Goo | ose
        ii) Duck -> Duc | k -> Duc | uck
        iii) Radius -> Rad | ius
        iv) Octopus -> Oct | opu | pus
    '''
    length_noun = len(noun)
    if length_noun < PHONE_LENGTH:
        return (noun + (PHONE_LENGTH - length_noun) * EMPTY_CHARACTER)

    last_phoneme_index = PHONE_LENGTH * math.floor(length_noun / PHONE_LENGTH)
    representation = noun[: last_phoneme_index]
    right_representation = noun[last_phoneme_index:]
    deltaL = (PHONE_LENGTH - length_noun % PHONE_LENGTH) % PHONE_LENGTH

    middle_representation = ''
    while deltaL > 0:
        middle_representation += noun[last_phoneme_index - deltaL]
        deltaL -= 1

    return representation + middle_representation + right_representation


def tensor_representation(noun: str) -> torch.Tensor:
    '''
    Converts a string into its associated tensor representation by mapping
    each character of its phonetic representation into a predefined index,
    then reshaping the tensor so that each row has PHONE_LENGTH columns
    '''
    phone: str = phonetic_representation(noun)
    string_to_index = convert_string_to_index(phone, letter_to_number)

    # by construction, PHONE_LENGTH divides the length of string_to_index
    index_length = len(string_to_index) // PHONE_LENGTH
    p = torch.tensor(string_to_index)
    return torch.reshape(p, (index_length, PHONE_LENGTH))


def phonological_data(nouns: 'list[str]', is_regular: bool) -> pd.DataFrame:
    '''
    Creates the data for the phonological representation of words that will
    be inputted into the model for training and testing
    '''
    plural_engine = inflect.engine()
    columns = ['noun', "noun_IPA", 'pluralized_noun_IPA', 'tensor_noun']
    rows = []
    for noun in nouns:
        noun = noun.translate(str.maketrans('', '', punctuation))
        noun_IPA = word_IPA(noun)
        pluralized_noun_IPA = word_IPA(plural_engine.plural_noun(noun))
        if len(noun_IPA) > 0 and len(pluralized_noun_IPA) > 0 and len(noun) > PHONE_LENGTH:
            tensor_noun = tensor_representation(noun)
            rows.append({'noun': noun,
                         'noun_IPA': noun_IPA,
                         'pluralized_noun_IPA': pluralized_noun_IPA,
                         'tensor_noun': tensor_noun})

    dataframe = pd.DataFrame(rows, columns=columns)
    dataframe = dataframe.set_index('noun')
    dataframe.to_csv(get_name(is_regular, True), sep='\t')
    return dataframe
