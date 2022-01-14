import json
import inflect
import os

import pandas as pd
import matplotlib.pyplot as plt

from data import find_all_singular_nouns
from similar import get_similar_words, predict_plural
from word_utils import *

COG_MODEL_NAME = "cognitive_model_epsilon=%d.obj"


def cognitive_model(nouns: 'list[str]', epsilon: int, engine: inflect.engine,
                    is_phon_model: bool, limit=25, save=True) -> pd.DataFrame:
    '''
    Runs the similarity computations to attain a data representation of
    a noun, the similar words, its predicted plural, its actual plural, and
    whether the prediction was accurate
    '''
    rows = []
    data, phoneme_dict = process_nouns(nouns=nouns,
                                       is_phon_model=is_phon_model,
                                       engine=engine)
    for noun in nouns:
        noun_rep, plural_rep = create_rep(noun=noun,
                                          engine=engine,
                                          is_phon_model=is_phon_model)
        if noun_rep and plural_rep:
            sim_words = get_similar_words(noun_rep=noun_rep,
                                          nouns=data,
                                          epsilon=epsilon,
                                          limit=limit)

            predicted_plural = predict_plural(noun_rep=noun_rep,
                                              sim_words=sim_words,
                                              engine=engine,
                                              is_phon_model=is_phon_model,
                                              phoneme_dict=phoneme_dict)
            rows.append({'noun': noun,
                         'noun_rep': noun_rep,
                         'similar_words': sim_words,
                         'predicted_plural': predicted_plural,
                         'actual_plural': plural_rep,
                         'accurate_prediction': plural_rep == predicted_plural})

    dataframe = pd.DataFrame(rows, columns=MODEL_COLUMNS)

    if save:
        filename = COG_MODEL_NAME % (epsilon)
        save_object(name=filename,
                    object=dataframe,
                    is_phon_model=is_phon_model,
                    folder_name=MODEL_FOLDER_NAME)

    return dataframe


def orthographic_model(nouns: 'list[str]', epsilon: int, engine: inflect.engine):
    '''
    Applies the orthographic model on the list of nouns
    '''
    return cognitive_model(nouns, epsilon, engine, is_phon_model=False)


def phonological_model(nouns: 'list[str]', epsilon: int, engine: inflect.engine):
    '''
    Applies the phonological model on the list of nouns
    '''
    return cognitive_model(nouns, epsilon, engine, is_phon_model=True)


def accuracy_values(nouns: 'list[str]', is_phon_model: bool,
                    engine: inflect.engine, max_epsilon: int):
    '''
    Returns a model's accuracy values for the model for each epsilon in 1, ..., max_epsilon - 1
    '''
    valuesX, valuesY = [], []

    for epsilon in range(1, max_epsilon):
        valuesX.append(epsilon)
        model = cognitive_model(nouns=nouns,
                                epsilon=epsilon,
                                engine=engine,
                                is_phon_model=is_phon_model,
                                save=False)

        output = model.accurate_prediction.value_counts()
        values = json.loads(output.to_json())
        accuracy_rate = calculate_accuracy_rate(values)
        valuesY.append(accuracy_rate)

    return valuesX, valuesY


def run_simulations(file_name: str, max_epsilon: int, engine: inflect.engine):
    '''
    Plots the models' accuracy rates over regular/irregular nouns and epsilon
    '''
    reg_nouns, irreg_nouns = find_all_singular_nouns(file_name, engine)
    all_nouns = irreg_nouns + reg_nouns

    print("Now running the models...")

    orthX, orthY = accuracy_values(nouns=all_nouns,
                                   is_phon_model=False,
                                   engine=engine,
                                   max_epsilon=max_epsilon)

    phonX, phonY = accuracy_values(nouns=all_nouns,
                                   is_phon_model=True,
                                   engine=engine,
                                   max_epsilon=max_epsilon)

    print("Now plotting the values...")

    fig, axs = plt.subplots(2)
    axs[0].plot(orthX, orthY, 'tab:green'), axs[0].set_title('O(Nouns)')
    axs[1].plot(phonX, phonY, 'tab:red'), axs[1].set_title('P(Nouns)')

    for ax in axs.flat:
        ax.set_ylim(bottom=0, top=1)
        ax.set(xlabel='Epsilon', ylabel='Accuracy Rate')

    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(os.path.join('Plots', 'Main_Plot.png'))


if __name__ == '__main__':
    epsilon, file_name = 3, 'big.txt'
    engine = inflect.engine()

    reg_nouns, irreg_nouns = find_all_singular_nouns(file_name)
    orth_regular = orthographic_model(nouns=reg_nouns,
                                      epsilon=epsilon,
                                      engine=engine)

    orth_irregular = orthographic_model(nouns=irreg_nouns,
                                        epsilon=epsilon,
                                        engine=engine)

    phon_regular = phonological_model(nouns=reg_nouns,
                                      epsilon=epsilon,
                                      engine=engine)

    phon_irregular = phonological_model(nouns=irreg_nouns,
                                        epsilon=epsilon,
                                        engine=engine)
