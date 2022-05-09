from collections import Counter
from string import punctuation
from typing import Dict, Sequence, Set, TypeVar
from mmda_pdf_scorer.utils import SentenceRows

from thefuzz import fuzz

Index = TypeVar('Index')


class BaseScorer:
    def build_index(self, rows: Sequence[SentenceRows]) -> Index:
        ...

    def score(self,
              query: SentenceRows,
              text: SentenceRows,
              index: Index) -> float:
        raise NotImplementedError()


class FuzzyScorer(BaseScorer):
    def score(self,
              query: SentenceRows,
              text: SentenceRows,
              index: Index) -> float:
        return fuzz.ratio(query.sentence, text.sentence) / 100.


class CtrlFScorer(BaseScorer):
    def __init__(self, lowercase: bool, punct: bool, ngrams: int = 0) -> None:
        self.lowercase = lowercase
        self.punct = punct
        self.ngrams = ngrams
        super().__init__()

    def build_index(self, rows: Sequence[SentenceRows]) -> Dict[str, Counter]:
        # we return an empty index we will use as cache
        return {}

    def transform_symbols(self, symbols: Sequence[str]) -> str:
        symbols = ''.join(symbols)
        if not self.punct:
            symbols = symbols.strip(punctuation)
        if self.lowercase:
            symbols = symbols.lower()
        return symbols

    def sentence_rows_to_counter(self, row: SentenceRows) -> Counter:
        sentence = tuple(self.transform_symbols(w.symbols) for w in row.words)

        if self.ngrams > 0:
            sentence = ' '.join(sentence)
            if len(sentence) >= self.ngrams:
                sentence = tuple(sentence[i: i + self.ngrams] for i in
                                 range(len(sentence) - self.ngrams + 1))
            else:
                spacing = (self.ngrams + len(sentence)) * ' '
                sentence = (spacing + sentence, sentence + spacing)

        c = Counter(sentence)

        # in case one word gets reduced to nothing
        c.pop(None, None)
        c.pop('', None)

        return c

    def score(self,
              query: SentenceRows,
              text: SentenceRows,
              index: Dict[str, Set[str]]) -> float:

        query_tokens = index.setdefault(
            query.uuid, self.sentence_rows_to_counter(query)
        )
        text_tokens = index.setdefault(
            text.uuid, self.sentence_rows_to_counter(text)
        )

        match_count = sum(cnt for tok, cnt in query_tokens.items()
                          if tok in text_tokens)

        return match_count / sum(query_tokens.values())
