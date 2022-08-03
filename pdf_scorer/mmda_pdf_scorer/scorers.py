from typing import Sequence, TypeVar
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
