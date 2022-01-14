import inflect

from word_utils import EMPTY_CHARACTER, DELETE, contains_empty_characters
from collections import Counter


def are_similar(wordA: str, wordB: str, epsilon: int or float):
    '''
    Checks if wordA and wordB are similar by checking if d(wordA, wordB) <= epsilon,
    where d: L x L -> R are Levenshtein distance metrics. We want nonidentical
    strings in the computation.
    '''
    dL = levenshtein_distance(wordA, wordB, epsilon)
    return 0 < dL <= epsilon


def levenshtein_distance(wordA: str, wordB: str, epsilon: int) -> int:
    '''
    Returns the Levenshtein distance between two strings, which is the minimum
    number of character edits (insertion, deletion, or substitutions)
    to change one word into the other (https://bit.ly/3ouLYcd)

    If the distance exceeds epsilon, we stop computing. It'll reduce the overall
    complexity when iterating through all words.
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


def attain_plural(noun_sing: str, engine: inflect.engine,
                  is_phon_model: bool, phoneme_dict: dict):
    '''
    Returns a word or phoneme's plural using either Inflect's plural_noun function,
    or the phoneme_dict.
    '''
    return phoneme_dict[noun_sing] if is_phon_model else engine.plural_noun(noun_sing)


def string_change(noun_sing: str, engine: inflect.engine, is_phon_model: bool,
                  phoneme_dict: dict):
    '''
    Returns a string that displays how noun_sing changed when pluralized.
    If the singular noun is shorter, it will insert a delete character that
    signifies the shortening.

    For example, if oxen goes to ox, then the change will be representated as
    **-- since "ox-" remains yet "-en" is deleted. Similarly, if duck goes to
    ducks, then the change would be represented as ****s
    '''
    noun_plural = attain_plural(noun_sing=noun_sing,
                                engine=engine,
                                is_phon_model=is_phon_model,
                                phoneme_dict=phoneme_dict)

    sing_length, plural_length = len(noun_sing), len(noun_plural)
    if noun_sing == noun_plural:
        return EMPTY_CHARACTER * len(noun_sing)

    # iterates through and checks the differences between noun_sing and noun_plural
    string_change = "*"
    iterating_index = min(sing_length, plural_length)
    for index in range(1, iterating_index):
        if noun_sing[index].lower() == noun_plural[index].lower():
            string_change += EMPTY_CHARACTER
        else:
            string_change += noun_plural[index].lower()

    if plural_length >= sing_length:
        string_change += noun_plural[sing_length:]
    else:
        string_change += DELETE * abs(plural_length - sing_length)

    return string_change


def formatted_string_change(word: str, word_orig: str, engine: inflect.engine,
                            is_phon_model: bool, phoneme_dict: dict):
    '''
    Formats the change in string to conform to the original word's structure
    '''
    change = string_change(noun_sing=word_orig,
                           engine=engine,
                           is_phon_model=is_phon_model,
                           phoneme_dict=phoneme_dict)

    change = change.replace(DELETE, '')

    word_diff = len(word) - len(word_orig)
    if word_diff >= 0:
        return word_diff * EMPTY_CHARACTER + change

    # removes the first i CHARACTERs and returns this change
    i = 0
    while i < abs(word_diff) and contains_empty_characters(change[i]):
        i += 1
    return change[i:]


def predict_plural(noun_rep: str, sim_words: 'list[str]', engine: inflect.engine,
                   is_phon_model: bool, phoneme_dict: dict) -> str:
    '''
    Predicts the plural of a noun by finding the most frequent plural change
    of words and applying it to noun
    '''
    # if there are no similar words, then the default is to regularize the noun
    if not sim_words:
        return pluralize_regular_noun(noun_rep, is_phon_model)

    changes_sim_words = Counter([
        formatted_string_change(noun_rep, word, engine, is_phon_model, phoneme_dict) for word in sim_words
    ])

    most_common_change, _ = changes_sim_words.most_common(1)[0]
    return apply_change_to_string(noun_rep, most_common_change)


def pluralize_regular_noun(noun: str, is_phon_model: bool):
    '''
    Pluralizes noun following the rules for regular nouns
    '''
    if not is_phon_model:
        if noun[-1] in ['s', 'x', 'z'] or noun[-2:] in ['ss', 'sh', 'ch']:
            return noun + 'es'
        return noun + 's'
    else:
        if noun[-1] in ['t', 'k', 'p']:
            return noun + 's'
        if noun[-1] == 's':
            return noun + 'iz'
        return noun + 'z'


def apply_change_to_string(noun: str, formatted_change: str):
    '''
    Applies formatted change to noun.
    '''

    changed_string = ''
    for index, changed_char in enumerate(formatted_change):
        if contains_empty_characters(changed_char):
            changed_string += noun[index].lower()
        else:
            changed_string += changed_char.lower()
    return changed_string


def get_similar_words(noun_rep: str, nouns: 'list[str]', epsilon: int,
                      limit: int) -> 'list[str]':
    '''
    Finds similar words within the array of words
    '''
    similar_words = []
    for other_word in nouns:
        if len(similar_words) <= limit:
            if are_similar(noun_rep, other_word, epsilon):
                similar_words.append(other_word)
        else:
            break
    return similar_words
