import re
from typing import List

import numpy as np
import spacy
from pdf2sents.typed_predictors import TypedBlockPredictor
from snorkel.labeling import LabelingFunction, LFAnalysis, LFApplier, labeling_function
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from snorkel.preprocess.nlp import SpacyPreprocessor
from spaczz.matcher import FuzzyMatcher

from weak_label.types import Instance
from weak_label.word_bank import WordBank

ABSTAIN = -1
NONSIG = 0
SIG = 1


class Annotator:
    nlp = spacy.blank("en")
    spacy_preprocessor = SpacyPreprocessor(
        text_field="text", doc_field="doc", memoize=True, memoize_key=id
    )
    wordbank = WordBank()

    def _keyword_lookup(x, keywords, label):
        if any(word in x.text.lower() for word in keywords):
            return label
        return ABSTAIN

    def _make_keyword_lf(keywords, label):
        return LabelingFunction(
            name=f"keyword_{keywords[0]}",
            f=Annotator._keyword_lookup,
            resources=dict(keywords=keywords, label=label),
        )

    def _find_section_matches(x: Instance, sections: List[str]):
        matcher = FuzzyMatcher(Annotator.nlp.vocab)
        for section in sections:
            matcher.add(section, [Annotator.nlp(section)])
        matches = matcher(Annotator.nlp(x.section))
        return matches

    @labeling_function()
    def contains_author_statement(x: Instance) -> int:
        pattern = r"(?=\b(" + "|".join(Annotator.wordbank.AUTHOR_KWS) + r")\b)"
        has_author_statement = re.search(pattern, x.text.lower())

        matches = Annotator._find_section_matches(x, ["acknowledgements"])
        if matches:
            return NONSIG

        if any(word in x.text.lower() for word in ["http"]):
            return NONSIG

        if any(word in x.text.lower() for word in ["arxiv"]):
            return NONSIG

        return SIG if has_author_statement else NONSIG

    @labeling_function(pre=[spacy_preprocessor])
    def exceeds_min_length(x: Instance) -> int:
        return NONSIG if len(x.doc) <= 5 else ABSTAIN

    @labeling_function(pre=[spacy_preprocessor])
    def exceeds_max_length(x: Instance) -> int:
        return NONSIG if len(x.doc) >= 80 else ABSTAIN

    @labeling_function(pre=[spacy_preprocessor])
    def contains_method_statement(x: Instance) -> int:
        ...

    @labeling_function(pre=[spacy_preprocessor])
    def contains_result_statement(x: Instance) -> int:
        sections = [
            "abstract",
            "introduction",
            "experiment",
            "result",
            "discussion",
            "conclusion",
        ]
        matches = Annotator._find_section_matches(x, sections)
        if not matches:
            return ABSTAIN
        for token in x.doc:
            if token.lemma_ in Annotator.wordbank.RESULT_KWS:
                return SIG
        return ABSTAIN

    @labeling_function(pre=[spacy_preprocessor])
    def contains_objective_statement(x: Instance) -> int:
        sections = [
            "abstract",
            "introduction",
            "background",
            "related work",
            "recent work",
            "conclusion",
        ]
        matches = Annotator._find_section_matches(x, sections)
        if not matches:
            return ABSTAIN
        for token in x.doc:
            if token.lemma_ in Annotator.wordbank.OBJECTIVE_KWS:
                return SIG
        return ABSTAIN

    @labeling_function(pre=[spacy_preprocessor])
    def contains_novelty_statement(x: Instance) -> int:
        # Novelty statements should be in an expected section
        sections = [
            "abstract",
            "introduction",
            "background",
            "related work",
            "recent work",
            "conclusion",
        ]
        matches = Annotator._find_section_matches(x, sections)
        if not matches:
            return ABSTAIN

        # Novelty statements should not include external reference
        if any(word in x.text.lower() for word in ["et al"]):
            return ABSTAIN
        citation_pattern = r"\(19[0-9]\d|20[0-9]\d|2100\)"
        matches = re.search(citation_pattern, x.text.lower())
        if matches:
            return ABSTAIN

        for token in x.doc:
            if token.lemma_ in Annotator.wordbank.NOVELTY_KWS:
                return SIG
        return ABSTAIN

    @labeling_function()
    def is_in_acknowledgements(x: Instance) -> int:
        sections = ["acknowledgements"]
        matches = Annotator._find_section_matches(x, sections)
        return NONSIG if matches else ABSTAIN

    @labeling_function()
    def contains_link(x: Instance):
        lf = Annotator._make_keyword_lf(keywords=["http"], label=NONSIG)
        return lf(x)

    @labeling_function()
    def contains_arxiv(x: Instance):
        lf = Annotator._make_keyword_lf(keywords=["arxiv"], label=NONSIG)
        return lf(x)

    @labeling_function()
    def is_list_elem_in_introduction(x: Instance) -> int:
        if x.type != TypedBlockPredictor.ListType:
            return ABSTAIN
        matcher = FuzzyMatcher(Annotator.nlp.vocab)
        matcher.add("intro", [Annotator.nlp("introduction")])
        matches = matcher(Annotator.nlp(x.section))
        return SIG if matches else ABSTAIN

    def apply_labeling_functions(self, dataset, lfs: List[LabelingFunction]):
        applier = LFApplier(lfs=lfs)
        return applier.apply(dataset)

    def get_lf_summary(self, labels: np.ndarray, lfs: List[LabelingFunction]):
        return LFAnalysis(L=labels, lfs=lfs).lf_summary()

    def predict(self, labels: np.ndarray, agg_model: str = "majority"):
        if agg_model == "majority":
            model = MajorityLabelVoter()
            return model.predict(L=labels)
        elif agg_model == "label":
            model = LabelModel(cardinality=2, verbose=True)
            model.fit(L_train=labels, n_epochs=500, log_freq=100, seed=69)
            return model.predict_proba(labels)
