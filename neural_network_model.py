import os
import pandas as pd


from sklearn.model_selection import train_test_split
from data import find_all_singular_nouns, get_file_pathname
from neural_network import run_neural_network
from model import cognitive_model


epsilon = 2
LIMIT = 10


def read_load_data(is_phonology: bool, file_name: str, epsilon=epsilon,
                   limit=LIMIT) -> pd.DataFrame:
    '''
    Reads and loads the data depending on it being a specific phonological
    dataset or orthographical. Rather than waste valueable time and memory 
    when rerunning this script, it will check if there is a csv file that 
    contains the data and load it into a pandas dataframe.
    '''

    regular_path = get_file_pathname(True, is_phonology, file_name)
    irregular_path = get_file_pathname(False, is_phonology, file_name)

    # if the file exists, we just need to load it using pandas
    if os.path.isfile(regular_path) and os.path.isfile(irregular_path):
        reg_noun_data = pd.read_csv(regular_path, sep='\t')
        irreg_noun_data = pd.read_csv(irregular_path, sep='\t')
        return reg_noun_data, irreg_noun_data

    else:
        reg_nouns, irreg_nouns = find_all_singular_nouns(file_name)
        print("Both Regular and Irregular nouns have been processed...\n\n")
        if not os.path.isfile(regular_path):
            reg_noun_data = load_data(reg_nouns, is_phonology, True, epsilon,
                                      limit, file_name)
        else:
            reg_noun_data = pd.read_csv(regular_path, sep='\t')

        if not os.path.isfile(irregular_path):
            irreg_noun_data = load_data(irreg_nouns, is_phonology, False, epsilon,
                                        limit, file_name)
        else:
            irreg_noun_data = pd.read_csv(irregular_path, sep='\t')

        return reg_noun_data, irreg_noun_data


def load_data(nouns: 'list[str]', is_phonology: bool, is_regular: bool, epsilon: int, limit: int, file_name: str):
    '''
    Queries the data from the nouns and outputs a formatted dataframe
    '''

    return cognitive_model(nouns, epsilon, limit)


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
    dataframes and perform a test, train, and split there.

    This will return the following:

    mix = False:
        (X_reg_train, X_reg_test, y_reg_train, y_reg_test), 
        X_irreg_train, X_irreg_test, y_irreg_train, y_irreg_test)

    mix = True:
        X_mixed_train, X_mixed_test, y_mixed_train, y_mixed_test
    '''
    reg_noun_data, irreg_noun_data = read_load_data(is_phonology, text_file)
    # cleans the data by ensuring the inputs are unique and singular
    X_reg, y_reg = extract_columns(reg_noun_data, -2, -1)
    X_irreg, y_irreg = extract_columns(irreg_noun_data, -2, -1)
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


def run_model(is_phonology: bool, test_size: int,
              text_file: str, mix=False):

    if not mix:
        reg, irreg = prepare_data(is_phonology, test_size, text_file, mix)
        print("Now printing the neural network's results for regular nouns\n\n")
        run_neural_network(reg)
        print("Now printing the neural network's results for regular nouns\n\n")
        run_neural_network(irreg)

    else:
        X_train, X_test, y_train, y_test = prepare_data(is_phonology, test_size,
                                                        text_file, mix)
        run_neural_network(X_train, X_test, y_train, y_test)
