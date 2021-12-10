from data import find_all_singular_nouns
import model

reg_nouns, irreg_nouns = find_all_singular_nouns('big.txt')

df = model.orthographic_model(reg_nouns, 3, True)
dfA = model.orthographic_model(irreg_nouns, 3, False)

df = model.phonological_model(reg_nouns, 3, True)
dfA = model.phonological_model(irreg_nouns, 3, False)
