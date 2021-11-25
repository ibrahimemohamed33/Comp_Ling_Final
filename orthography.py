import pandas as pd
import inflect

CHARACTER = "*"


def hamming_distance(wordA: str, wordB: str) -> int:
    '''
    Computes the Hamming distance between two strings. This is particularly useful
    in analyzing similar words that are of the same length, yet can differ more than
    the levenshtein distance (e.g., foot and feet, goose and geese)
    '''
    N, count = len(wordA), 0
    if wordA == wordB or N != len(wordB):
        return 0

    for index in range(N):
        count += int(wordA[index] != wordB[index])
    return count


def levenshtein_distance(wordA: str, wordB: str) -> int:
    '''
    Returns the Levenshtein distance of two strings, which is the minimum
    number of character edits (insertion, deletion, or substitutions) 
    to change one word into the other

    The explicit formula can be found in the wikipedia article: https://bit.ly/3r1NHYf
    '''
    if len(wordA) == 0:
        return len(wordB)
    if len(wordB) == 0:
        return len(wordA)

    tailA, tailB = wordA[1:], wordB[1:]
    if wordA[0] == wordB[0]:
        return levenshtein_distance(tailA, tailB)

    minimum_distance = min(levenshtein_distance(tailA, wordB),
                           levenshtein_distance(wordA, tailB),
                           levenshtein_distance(tailA, tailB))

    return 1 + minimum_distance


def string_change(word_singular: str, word_plural: str):
    '''
    Returns a string that displays how the singular version of a word changed
    '''
    if word_singular == word_plural:
        return CHARACTER * len(word_singular)

    lengthA, lengthB = len(word_singular), len(word_plural)
    length_min, length_max = min(lengthA, lengthB), max(lengthA, lengthB)
    word_max, word_min = word_singular, word_plural if lengthA == length_max else word_plural, word_singular

    # discrepancy = ''
    # for index in len(word_min):
    #     if string[index] ==


def similar_pluralized(word: str, words: 'list[str]', engine: inflect.engine,
                       epsilon: int, limit=10) -> 'list[str]':
    similar_words = []
    for other_word in words:
        if len(similar_words) <= limit:
            distance_levenshtein = levenshtein_distance(word, other_word)
            distance_hamming = hamming_distance(word, other_word)
            if distance_levenshtein < epsilon or distance_hamming <= epsilon:
                pluralized_word = engine.plural_noun(word)

                similar_words.append(pluralized_word)
        else:
            break

    return similar_words


def orthographical_data(nouns, epsilon: int, limit=10) -> pd.DataFrame:
    engine = inflect.engine()
    rows = []
    for noun in nouns:
        similar_words = similar_pluralized(noun, epsilon, limit)
        pluralized_noun = engine.plural_noun(noun)
        values = {
            'noun': noun,
            'similar_words': similar_words,
            'pluralized_noun': pluralized_noun
        }
        rows.append(values)

    columns = ["noun", "similar_words", 'pluralized_noun']
    return pd.DataFrame(rows, columns=columns)
