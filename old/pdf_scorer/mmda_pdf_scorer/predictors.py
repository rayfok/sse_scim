from tempfile import NamedTemporaryFile
from mmda.predictors.heuristic_predictors.dictionary_word_predictor \
    import DictionaryWordPredictor

import datasets

DICTIONARY = 'https://github.com/dwyl/english-words/raw/master/words_alpha.zip'


def get_dictionary_word_predictor():
    # takes care of downloading
    words = datasets.load_dataset("text", data_files=DICTIONARY)

    # write to temp file, then create a dictionary word predictor from there
    with NamedTemporaryFile(mode='w') as f:
        f.writelines(words['train']['text'])
        f.close()
        predictor = DictionaryWordPredictor(f.name)

    return predictor
