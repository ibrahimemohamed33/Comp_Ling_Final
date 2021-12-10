import json
import inflect
import os

import pandas as pd
import matplotlib.pyplot as plt

from data import find_all_singular_nouns
from similar import get_similar_words, predict_plural
from word_utils import MODEL_COLUMNS, create_rep, process_nouns, save_object

COG_MODEL_NAME = "cognitive_model_epsilon=%d.obj"


def cognitive_model(nouns, epsilon: int, is_regular: bool, limit=25,
                    is_phonology=False, save=True) -> pd.DataFrame:
    '''
    Runs the similarity computations to attain a data representation of
    a noun, the similar words, its predicted plural, its actual plural, and
    whether the prediction was accurate
    '''
    rows, engine = [], inflect.engine()
    data, phoneme_dict = process_nouns(nouns, is_phonology, engine, is_regular)

    for noun in nouns:
        noun_rep, plural_rep = create_rep(noun, engine, is_phonology)
        if noun_rep and plural_rep:
            sim_words = get_similar_words(noun_rep, data, epsilon, limit)
            predicted_plural = predict_plural(noun_rep, sim_words, engine,
                                              is_phonology, phoneme_dict)
            rows.append({'noun': noun,
                         'noun_rep': noun_rep,
                         'similar_words': sim_words,
                         'predicted_plural': predicted_plural,
                         'actual_plural': plural_rep,
                         'accurate_prediction': plural_rep == predicted_plural})

    dataframe = pd.DataFrame(rows, columns=MODEL_COLUMNS)

    if save:
        save_object(COG_MODEL_NAME % (epsilon), is_regular, dataframe,
                    is_phonology, folder_name="saved_models")

    return dataframe


def orthographic_model(nouns: 'list[str]', epsilon: int, is_regular: bool):
    return cognitive_model(nouns, epsilon, is_regular, is_phonology=False)


def phonological_model(nouns: 'list[str]', epsilon: int, is_regular: bool):
    return cognitive_model(nouns, epsilon, is_regular, is_phonology=True)


def accuracy_values(nouns: 'list[str]', is_regular: bool, is_phonology: bool,
                    max_epsilon: int, debug=True):
    '''
    Returns the accuracy values for a the model for each 
    epsilon in 1, ..., max_epsilon - 1
    '''

    x_values, y_values = [], []
    for epsilon in range(1, max_epsilon):
        x_values.append(epsilon)
        df = cognitive_model(nouns, epsilon, is_regular,
                             25, is_phonology, False)

        values = json.loads(df.accurate_prediction.value_counts().to_json())
        if debug:
            print("values=", values)
            pd.set_option("display.max_rows", 100, "display.max_columns", None)
        try:
            accuracy_rate = values['True'] / (values['True'] + values['False'])
        except KeyError as e:
            accuracy_rate = 0 if e == "True" else 1

        y_values.append(accuracy_rate)

    return x_values, y_values


def run_simulations(file_name: str, max_epsilon: int):
    '''
    Runs the cognitive models and plots the accuracy rates over regular/irregular noun
    and epsilon
    '''
    reg_nouns, irreg_nouns = find_all_singular_nouns(file_name)

    print("Now running the models on irregular nouns ")

    p2, q2 = accuracy_values(irreg_nouns, False, False, max_epsilon)
    x2, y2 = accuracy_values(irreg_nouns, False, True, max_epsilon)

    print("Now running the models on regular nouns ")
    p1, q1 = accuracy_values(reg_nouns, True, False, max_epsilon)
    x1, y1 = accuracy_values(reg_nouns, True, True, max_epsilon)

    print("Now plotting the values")
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(p1, q1), axs[0, 0].set_title('O(Regular Nouns)')
    axs[1, 0].plot(p2, q2, 'tab:green'), axs[1,
                                             0].set_title('O(Irregular Nouns)')

    axs[0, 1].plot(x1, y1, 'tab:orange'), axs[0,
                                              1].set_title('P(Regular Nouns)')
    axs[1, 1].plot(x2, y2, 'tab:red'), axs[1,
                                           1].set_title('P(Irregular Nouns)')

    for ax in axs.flat:
        ax.set_ylim(bottom=0, top=1)
        ax.set(xlabel='Epsilon', ylabel='Accuracy Rate')

    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(os.path.join('Plots', 'Main_Plot.png'))
