import inflect
import pandas as pd
import eng_to_ipa as ipa
import numpy as np

from string import punctuation
PHONE_LENGTH = 3
EMPTY_CHARACTER = "*"


def word_IPA(word: str) -> str:
    '''
    Checks if a given word is in the CMU dictionary and then returns its
    corresponding IPA
    '''
    return ipa.convert(word) if ipa.isin_cmu(word) else EMPTY_CHARACTER


def are_nouns_OK(nouns):
    for noun in nouns:
        if len(noun) == 0 or noun == EMPTY_CHARACTER:
            return False
    return True


def noun_data(noun: str, engine: inflect.engine):
    '''
    Given a noun, returns the necessary data (its plural as well as the IPAs)
    '''
    sing_noun = noun.translate(str.maketrans('', '', punctuation)).lower()
    plural_noun = engine.plural_noun(sing_noun).lower()
    noun_IPA, plural_IPA = word_IPA(sing_noun), word_IPA(plural_noun)

    values = [sing_noun, plural_noun, noun_IPA, plural_IPA]
    return tuple([" ".join(string) for string in values])


def phonological_data(nouns: 'list[str]', is_regular: bool,
                      file_name: str) -> pd.DataFrame:
    '''
    Creates the data for the phonological representation of words that will
    be inputted into the model for training and testing
    '''
    engine = inflect.engine()
    columns = ['noun', 'plural']
    # columns = ['noun', "plural", "noun_IPA", 'plural_noun_IPA']

    rows = []
    for noun in nouns:
        # formats the singular noun to prevent any errors in punctuation or casing
        noun, plural_noun, noun_IPA, plural_noun_IPA = noun_data(noun, engine)
        if are_nouns_OK([noun, plural_noun, noun_IPA, plural_noun_IPA]):
            rows.append({"noun": np.asarray(noun).astype(str),
                         "plural": np.asarray(plural_noun).astype(str)})

    dataframe = pd.DataFrame(rows, columns=columns)
    # dataframe = dataframe.set_index('noun')
    # dataframe.to_csv(get_name(is_regular, True, file_name), sep='\t')
    return dataframe
