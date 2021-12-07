import json
import inflect
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

from random import shuffle
from collections import Counter
from data import find_all_singular_nouns
from phonology import are_nouns_OK, word_IPA

EMPTY_CHARACTER, DELETE = "*", '-'
PHONEME_DICT_NAME = "phoneme_to_phoneme_plural.obj"


def are_similar(wordA: str, wordB: str, epsilon: int or float):
    '''
    Checks if wordA and wordB are similar by checking if d(wordA, wordB) <= epsilon,
    where d: L x L -> R are Levenshtein and Hamming distance metrics. Since we 
    want different words from wordA or wordB, we must discount identical strings

    The Hamming distance serves as an upper bound to the Levenshtein distance
    for strings of the same length
    '''
    distance_levenshtein = levenshtein_distance(wordA, wordB, epsilon)
    distance_hamming = hamming_distance(wordA, wordB)
    return 0 < distance_levenshtein <= epsilon or 0 < distance_hamming <= epsilon


def hamming_distance(wordA: str, wordB: str) -> int:
    '''
    Computes the Hamming distance between two strings. This is particularly useful
    in analyzing similar words that are of the same length, yet can differ more than
    the levenshtein distance (e.g., foot and feet, goose and geese)
    '''
    N, count = len(wordA), 0

    if N != len(wordB):
        return float("inf")

    if wordA == wordB:
        return 0

    for index in range(N):
        count += int(wordA[index].lower() != wordB[index].lower())
    return count


def levenshtein_distance(wordA: str, wordB: str, epsilon: int) -> int:
    '''
    Returns the Levenshtein distance of two strings, which is the minimum
    number of character edits (insertion, deletion, or substitutions)
    to change one word into the other.

    If the distance exceeds epsilon, we stop computing. It'll reduce the overall
    complexity when iterating through all words.

    The explicit formula can be found in the wikipedia article: https://bit.ly/3ouLYcd
    '''

    # there is no need to keep going if the distance is above epsilon
    if epsilon < 0:
        return float('inf')

    if not len(wordA):
        return len(wordB)
    if not len(wordB):
        return len(wordA)

    tailA, tailB = wordA[1:], wordB[1:]
    if wordA[0] == wordB[0]:
        return levenshtein_distance(tailA, tailB, epsilon)

    # if there is no match, then we decrement our epsilon to limit the number
    # of unnecessary checks since we only care about those with distances <= epsilon
    minimum_distance = min(levenshtein_distance(tailA, wordB, epsilon - 1),
                           levenshtein_distance(wordA, tailB, epsilon - 1),
                           levenshtein_distance(tailA, tailB, epsilon - 1))

    return 1 + minimum_distance


def attain_plural(noun_singular: str, engine: inflect.engine,
                  is_phonology: bool, phoneme_dict: dict):
    '''
    Returns the plural of a word or phoneme. It does so by leveraging Inflect's
    plural_noun function for orthographical words, or doing a lookup on phoneme_dict
    if we're finding the plural of a phoneme
    '''
    return phoneme_dict[noun_singular] if is_phonology else engine.plural_noun(noun_singular)


def string_change(noun_sing: str, engine: inflect.engine, is_phonology: bool,
                  phoneme_dict: dict):
    '''
    Returns a string that displays how noun_sing changed when pluralized.
    If the singular noun is shorter, it will insert a delete character that
    signifies the shortening.

    For example, if oxen goes to ox, then the change will be representated as
    **-- since "ox-" remains yet "-en" is deleted. Similarly, if duck goes to
    ducks, then the change would be represented as ****s
    '''

    noun_plural = attain_plural(noun_sing, engine, is_phonology, phoneme_dict)
    sing_length, plural_length = len(noun_sing), len(noun_plural)

    if noun_sing == noun_plural:
        return EMPTY_CHARACTER * len(noun_sing)

    # iterates through and checks the differences between noun_sing and noun_plural
    string_change = ""
    iterating_index = min(sing_length, plural_length)
    for index in range(iterating_index):
        if noun_sing[index].lower() == noun_plural[index].lower():
            string_change += EMPTY_CHARACTER
        else:
            string_change += noun_plural[index].lower()

    if plural_length >= sing_length:
        string_change += noun_plural[sing_length:]
    else:
        string_change += DELETE * abs(plural_length - sing_length)

    return string_change


def format_string_change(word: str, word_orig: str, engine: inflect.engine,
                         is_phonology: bool, phoneme_dict: dict):
    '''
    Formats the change in string to conform to the original word's structure
    '''
    change = string_change(word_orig, engine, is_phonology,
                           phoneme_dict).replace(DELETE, '')

    word_diff = len(word) - len(word_orig)
    if word_diff >= 0:
        return word_diff * EMPTY_CHARACTER + change

    # removes the first i CHARACTERs and then returns this change
    i = 0
    while i < abs(word_diff) and change[i] == EMPTY_CHARACTER:
        i += 1
    return change[i:]


def predict_plural(noun: str, sim_words: 'list[str]', engine: inflect.engine,
                   is_phonology: bool, phoneme_dict: dict) -> str:
    '''
    Predicts the plural of a noun by finding the most frequent plural change
    of words and applying it to noun
    '''
    # if there are no similar words, then the default is to regularize the noun
    if not len(sim_words):
        return pluralize_regular_noun(noun, is_phonology)

    changes_sim_words = Counter([
        format_string_change(noun, word, engine, is_phonology, phoneme_dict) for word in sim_words
    ])

    most_common_change, _ = changes_sim_words.most_common(1)[0]
    return apply_change_to_string(noun, most_common_change)


def pluralize_regular_noun(noun: str, is_phonology: bool):
    '''
    Pluralizes noun as if noun is regular
    '''
    if not is_phonology:
        if noun[-1] in ['s', 'x', 'z'] or noun[-2:] in ['ss', 'sh', 'ch']:
            return noun + 'es'
        return noun + 's'
    else:
        if noun[-1] in ['t', 'k', 'p']:
            return noun + 's'
        return noun + 'z'


def apply_change_to_string(noun: str, formatted_change: str):
    '''
    Applies the formatted change to noun. Since the length of the formatted
    change is within a small neighborhood of noun, the noun will be properly
    formatted. For example, if formatted_change = "****s", then it'll apply to
    the string "duck" and convert "duck" to "ducks"
    '''

    changed_string = ''
    for index in range(len(formatted_change)):
        if formatted_change[index] == EMPTY_CHARACTER:
            changed_string += noun[index].lower()
        else:
            changed_string += formatted_change[index].lower()
    return changed_string


def get_similar_words(word: str, words: 'list[str]', epsilon: int,
                      limit: int) -> 'list[str]':
    similar_words = []
    for other_word in words:
        if len(similar_words) <= limit:
            if are_similar(word, other_word, epsilon):
                similar_words.append(other_word)
        else:
            break
    return similar_words


def cog_model(nouns, epsilon: int, is_regular: bool, limit=25, is_phonology=False) -> pd.DataFrame:
    rows, engine = [], inflect.engine()
    data, phoneme_dict = process_nouns(nouns, is_phonology, engine, is_regular)
    columns = ["noun", 'noun_rep', 'similar_words', "predicted_plural",
               'actual_plural', 'accurate_prediction']

    for noun in nouns:
        noun_rep, plural_rep = create_rep(noun, engine, is_phonology)
        if noun_rep != EMPTY_CHARACTER and plural_rep != EMPTY_CHARACTER:
            sim_words = get_similar_words(noun_rep, data, epsilon, limit)
            predicted_plural = predict_plural(noun_rep, sim_words, engine,
                                              is_phonology, phoneme_dict)
            rows.append({'noun': noun,
                         'noun_rep': noun_rep,
                         'similar_words': sim_words,
                         'predicted_plural': predicted_plural,
                         'actual_plural': plural_rep,
                         'accurate_prediction': plural_rep == predicted_plural})

    return pd.DataFrame(rows, columns=columns)


def process_nouns(nouns: 'list[str]', is_phonology: bool, engine: inflect.engine,
                  is_regular: bool):
    if not is_phonology:
        return nouns, {}
    else:
        regular_name = "REGULAR" if is_regular else "IRREGULAR"
        file_name = regular_name + "_" + PHONEME_DICT_NAME
        phoneme_dict, phoneme_list = {}, []
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                phoneme_dict = pickle.load(f)

        for noun in nouns:
            plural_noun = engine.plural_noun(noun)
            noun_IPA, plural_IPA = word_IPA(noun), word_IPA(plural_noun)
            if noun_IPA != EMPTY_CHARACTER and plural_IPA != EMPTY_CHARACTER:
                phoneme_list.append(noun_IPA)
                if noun_IPA not in phoneme_dict:
                    phoneme_dict[noun_IPA] = plural_IPA

        with open(file_name, 'wb') as f:
            pickle.dump(phoneme_dict, f)

        return phoneme_list, phoneme_dict


def create_rep(noun: str, engine: inflect.engine, is_phonology: str):
    plural_noun = engine.plural_noun(noun)
    if not is_phonology:
        return noun, plural_noun
    return word_IPA(noun), word_IPA(plural_noun)


def accuracy_values(nouns: 'list[str]', is_regular: bool, is_phonology: bool, max_epsilon: int,
                    debug=True):

    x_values, y_values = [], []
    for epsilon in range(1, max_epsilon):
        x_values.append(epsilon)
        df = cog_model(nouns, epsilon, is_regular, is_phonology=is_phonology)
        values = json.loads(df.accurate_prediction.value_counts().to_json())
        if debug:
            print(epsilon)
            print(values)
            print(df)
            pd.set_option("display.max_rows", 20, "display.max_columns", None)

        try:
            accuracy_rate = values['True'] / (values['True'] + values['False'])
        except KeyError as e:
            accuracy_rate = 0 if e == "True" else 1

        y_values.append(accuracy_rate)

    return x_values, y_values


def run_models(file_name: str, max_epsilon: int):
    '''
    Runs the cognitive models and plots the accuracy rates over regular/irregular noun
    and epsilon
    '''
    reg_nouns, irreg_nouns = find_all_singular_nouns(file_name)

    print("Now running the models on irregular nouns ")
    p2, q2 = accuracy_values(irreg_nouns, False, is_phonology=False,
                             max_epsilon=max_epsilon)

    x2, y2 = accuracy_values(irreg_nouns, False, is_phonology=True,
                             max_epsilon=max_epsilon)

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
        ax.set(xlabel='Epsilon', ylabel='Accuracy Rate')

    for ax in axs.flat:
        ax.label_outer()

    plt.savefig('Main_Plot.png')


def generate_random_nouns(reg_nouns: 'list[str]', irreg_nouns: 'list[str]',
                          limit: int):
    shuffle(reg_nouns)
    shuffle(irreg_nouns)
    return reg_nouns[:limit], irreg_nouns[:limit]
