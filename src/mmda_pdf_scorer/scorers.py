from collections import Counter
from typing import Dict, Sequence, Set, TypeVar
from .utils import SentenceRows
from thefuzz import fuzz
import pylcs

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


class LongestCommonSubsequenceScorer(BaseScorer):
    def score(self,
              query: SentenceRows,
              text: SentenceRows,
              index: Index) -> float:
        return pylcs.lcs(query.sentence, text.sentence) / len(query.sentence)


class CtrlFScorer(BaseScorer):
    def __init__(self, lowercase: bool) -> None:
        self.lowercase = lowercase
        super().__init__()

    def build_index(self, rows: Sequence[SentenceRows]) -> Dict[str, Counter]:
        # we return an empty index we will use as cache
        return {}

    def transform_symbols(self, symbols: Sequence[str]) -> str:
        symbols = ''.join(symbols)
        if self.lowercase:
            symbols = symbols.lower()
        return symbols

    def sentence_rows_to_counter(self, row: SentenceRows) -> Counter:
        c = Counter(self.transform_symbols(w.symbols) for w in row.words)

        # in case one gets reduced to nothing
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
