# Pluralizing Unfamiliar Nouns

This repository contains the text files, code, and objects used in my final project for Computational Linguistics (CMSC 25610). The submitted paper can be found using this [link](https://www.dropbox.com/s/ue1r2dujkp3bwnz/Final%20Project%20Linguistics.pdf?dl=0). At the heart of the paper is the question of how English speakers pluralize nouns they have not been exposed to, like _moose_, _yeoman_, or _boose_. I found that following an approach of string-similarity proved to be a reliable method up to a limit of epsilon = 2. However, pluralizing nouns by their orthography consistently arrived at the right inflection as opposed to pluralizing by their phonemes.

## Running the Models
Once you have your text corpus and inputted it into the _Text_Files_ folder, you need to obtain the regular and irregular nouns of the corpus. But you need to first run your inflect engine
```python
import inflect
engine = inflect.engine()

from data import find_all_singular_nouns
nouns = find_all_singular_nouns(YOUR_FILENAME)
```

Once you have all your nouns, you can obtain a dataframe representing the nouns' orthographic or phonological results by running the commands:
```python
from model import orthographic_model, phonological_model

oModel = orthographic_model(nouns=nouns, epsilon=YOUR_EPSILON_VALUE, engine=engine)
phModel = phonological_model(nouns=nouns, epsilon=YOUR_EPSILON_VALUE, engine=engine)
```

To run the simulations and compare these models, simply run the commands
```python
from model import run_simulations

run_simulations(file_name=YOUR_FILE_NAME, max_epsilon=YOUR_MAX_EPSILON, engine=engine)
```

