import inflect

from word_utils import EMPTY_CHARACTER, DELETE
from collections import Counter


def are_similar(wordA: str, wordB: str, epsilon: int or float):
    '''
    Checks if wordA and wordB are similar by checking if d(wordA, wordB) <= epsilon,
    where d: L x L -> R are Levenshtein distance metrics. Since we 
    want different words from wordA or wordB, we must discount identical strings

    The Hamming distance serves as an upper bound to the Levenshtein distance
    for strings of the same length
    '''
    distance_levenshtein = levenshtein_distance(wordA, wordB, epsilon)
    return 0 < distance_levenshtein <= epsilon


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
    '''
    Finds similar words within the array of words
    '''
    similar_words = []
    for other_word in words:
        if len(similar_words) <= limit:
            if are_similar(word, other_word, epsilon):
                similar_words.append(other_word)
        else:
            break
    return similar_words
