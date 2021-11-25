import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data import find_all_singular_nouns, get_name
from orthography import orthographical_data
from phonology import phonological_data


epsilon = 2
LIMIT = 10


def load_data(nouns: 'list[str]', is_phonology: bool, is_regular: bool,
              epsilon: int, limit: int) -> pd.DataFrame:
    '''
    Queries the data from the nouns and outputs a formatted dataframe
    '''

    if is_phonology:
        return phonological_data(nouns, is_regular)
    else:
        return orthographical_data(nouns, epsilon, limit)


def read_load_data(is_phonology: bool, text_file: str, eps=epsilon,
                   limit=LIMIT) -> pd.DataFrame:
    '''
    Reads and loads the data depending on it being a specific phonological
    dataset or orthographical. Rather than waste valueable time and memory 
    when rerunning this script, it will check if there is a csv file that 
    contains the data and load it into a pandas dataframe.
    '''

    regular_path = get_name(True, is_phonology)
    irregular_path = get_name(False, is_phonology)

    # if the file exists, we just need to load it using pandas
    if os.path.isfile(regular_path) and os.path.isfile(irregular_path):
        reg_noun_data = pd.read_csv(regular_path, sep='\t')
        irreg_noun_data = pd.read_csv(irregular_path, sep='\t')
        return reg_noun_data, irreg_noun_data

    else:
        reg_nouns, irreg_nouns = find_all_singular_nouns(text_file)
        print("Both Regular and Irregular nouns have been processed...\n\n")
        if not os.path.isfile(regular_path):
            reg_noun_data = load_data(reg_nouns, is_phonology, True, eps,
                                      limit)
        else:
            reg_noun_data = pd.read_csv(regular_path, sep='\t')

        if not os.path.isfile(regular_path):
            irreg_noun_data = load_data(irreg_nouns, is_phonology, True,
                                        eps, limit)
        else:
            irreg_noun_data = pd.read_csv(regular_path, sep='\t')

        return reg_noun_data, irreg_noun_data


def extract_columns(dataframe: pd.DataFrame, columnA_index: int,
                    columnB_index: int):
    '''
    Extracts the column(s) from the given pandas dataframe
    '''
    columnA = dataframe[dataframe.columns[columnA_index]]
    columnB = dataframe[dataframe.columns[columnB_index]]
    return columnA, columnB


def prepare_data(is_phonology: bool, test_size: int, text_file: str, mix=False):
    '''
    Prepares the data for a test, train, and split by extracting the regular and
    irregular nouns separately if mix is false. Otherwise, it would combine the 
    dataframes and perform a test, train, and split there
    '''
    reg_noun_data, irreg_noun_data = read_load_data(is_phonology, text_file)
    # cleans the data by ensuring the inputs are unique and singular
    X_reg, y_reg = extract_columns(reg_noun_data, 0, 1)
    X_irreg, y_irreg = extract_columns(irreg_noun_data, 0, 1)
    if not mix:
        tts_reg = train_test_split(X_reg, y_reg, test_size=test_size,
                                   random_state=50, shuffle=True)

        tts_irreg = train_test_split(X_irreg, y_irreg, test_size=test_size,
                                     random_state=50, shuffle=True)

        return tts_reg, tts_irreg

    X_mixed = pd.concat([X_reg, X_irreg], ignore_index=False, sort=False)
    y_mixed = pd.concat([y_reg, y_irreg], ignore_index=False, sort=False)

    return train_test_split(X_mixed, y_mixed, test_size=test_size,
                            random_state=50, shuffle=True)


def extract_nlp_model(is_regular: bool, is_phonology: bool):
    return None


def compute_logistic_accuracy(model_output: np.array, expected_output: np.array):
    if len(model_output) != len(expected_output):
        raise Exception(
            f"The model's output array is of length {len(model_output)}"
            f"whereas the array of comparison has length {len(expected_output)}")

    N = len(model_output)
    total_count = 0
    for index in N:
        total_count += int(model_output[index] == expected_output[index])
    return total_count / N


def run_model(is_regular: bool, is_phonology: bool, test_size: int,
              text_file: str):

    trainX, trainY, testX, testY = prepare_data(is_regular, is_phonology,
                                                test_size, text_file)

    neural_network = extract_nlp_model()
    pass


# (X_train, y_train, _, _), (Xi_train, yi_train, _, _) = prepare_data(
#     True, 0.05, 'very_small.txt', mix=False)
# pd.set_option("display.max_rows", None, "display.max_columns", None)
# print('X_train\n\n', X_train)
# print('y_train\n\n', y_train)

# print('Xi_train\n\n', Xi_train)
# print('yi_train\n\n', yi_train)
