import eng_to_ipa as ipa
import string

from collections import Counter








alphabet = list(string.ascii_lowercase)
vowels = ['a', 'e', 'i', 'o', 'u']
consonants = [letter for letter in alphabet if letter not in vowels]


def is_vowel(letter: str) -> bool:
    return len(letter) == 1 and letter in vowels


def generate_pseudowords(suffix: str, num_letters: int):
    # suffixes that begin with a vowel should have a consonant preceding it
    formatted_suffix = suffix.replace('-', '')
    initial_vowel = is_vowel(suffix[0])
    letters = consonants if initial_vowel else vowels

    # we iterate through each letter and see if this suffix is a word in the
    # CMU dictionary. If not, then we generated a real-life pseudoword
    pseudowords = []
    for letter in letters:
        new_word = letter + formatted_suffix
        if not ipa.isin_cmu(new_word):
            pseudowords.append(new_word)
    return pseudowords


def create_rhyme_words(fake_word: str, num_letters: int, in_ipa=True):
    suffix = fake_word[-num_letters:]
    for letter in alphabet:
        new_word = letter + suffix
        if ipa.isin_cmu(new_word):
            rhymes = ipa.get_rhymes(new_word)
            if in_ipa:
                return [ipa.convert(rhyme) for rhyme in rhymes]
            return rhymes
    return []


def word_IPA(fake_word: str):
    if ipa.isin_cmu(fake_word):
        return ipa.convert(fake_word)
    

    # finding the word's IPA requires iterating through words that rhyme with
    # the fake word

    generated_rhymes = create_rhyme_words(fake_word, fake_word // 2)
    possible_suffixes = [rhyme[:]]
