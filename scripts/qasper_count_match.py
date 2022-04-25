import logging
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, Literal, NamedTuple, Sequence, Set

import datasets
import espresso_config as es
import spacy
from nltk.stem.porter import PorterStemmer
from spacy.tokens.span import Span
from spacy.util import filter_spans
from textacy.extract.basics import ngrams, noun_chunks


LOGGER = logging.getLogger(__file__)
LOGGER.setLevel(logging.INFO)


class MatchedChunk(NamedTuple):
    source: Span
    target: Span
    match: Span


@dataclass
class QAIntersections:
    abstract_question: Dict[Span, MatchedChunk] = \
        field(default_factory=dict)
    abstract_answer: Dict[Span, MatchedChunk] = \
        field(default_factory=dict)
    abstract_answer_question: Dict[Span, MatchedChunk] = \
        field(default_factory=dict)

    def __len__(self):
        return (len(self.abstract_question) +
                len(self.abstract_answer) +
                len(self.abstract_answer_question))

    def __bool__(self):
        return len(self) > 0


class Seq2SeqFeaturesMapperWithFocus:
    _TEXTACY_POS_EXPAND_NC = {'NOUN', 'VERB', 'PROPN'}
    _TEXTACY_POS_INITIAL_NC = {'NOUN', 'ADJ', 'VERB', 'PROPN'}
    _SPACY_DISABLE = ['ner', 'textcat']

    def __init__(
        self,
        expand_abstract: bool = False,
        expand_question: bool = False,
        expand_answer: bool = False,
        use_lemma: bool = False,
        use_stem: bool = False,
        spacy_model_name: str = "en_core_web_sm",
        spacy_disable: Sequence[str] = None,
        textacy_pos_initial_nc: Sequence[str] = None,
        textacy_pos_expand_nc: Sequence[str] = None,
        use_overlap_with_questions: bool = True,
        use_overlap_with_answers: bool = False,
        focus_strategy: Literal['wrap', 'after'] = 'wrap',
    ):
        self.focus_strategy = focus_strategy
        self.expand_abstract = expand_abstract
        self.expand_answer = expand_answer
        self.use_lemma = use_lemma
        self.use_stem = use_stem
        self.use_overlap_with_questions = use_overlap_with_questions
        self.use_overlap_with_answers = use_overlap_with_answers
        self.spacy_model_name = spacy_model_name
        self.expand_question = expand_question
        self.spacy_disable = spacy_disable or self._SPACY_DISABLE
        self.textacy_pos_expand_nc = set(
            textacy_pos_expand_nc or self._TEXTACY_POS_EXPAND_NC
        )
        self.textacy_pos_initial_nc = set(
            textacy_pos_initial_nc or self._TEXTACY_POS_INITIAL_NC
        )

    def get_spacy_pipeline(self):
        return spacy.load(self.spacy_model_name, disable=self.spacy_disable)

    def find_chunks(self, doc):
        extracted_noun_chunks = noun_chunks(doc, drop_determiners=True)
        for noun_chunk in extracted_noun_chunks:
            not_acceptable_start_word = True
            while not_acceptable_start_word and len(noun_chunk) > 0:
                not_acceptable_start_word = \
                    noun_chunk[0].pos_ not in self.textacy_pos_initial_nc
                noun_chunk = noun_chunk[1 if not_acceptable_start_word else 0:]
            if len(noun_chunk) > 0:
                yield noun_chunk

    def expand_chunks_set(self, extracted_noun_chunks):
        noun_chunk_ngram_subsets = set(extracted_noun_chunks)
        for noun_chunk in set(noun_chunk_ngram_subsets):
            allowed_pos_count = sum(
                1 if token.pos_ in self.textacy_pos_expand_nc
                else 0 for token in noun_chunk
            )
            if allowed_pos_count == 0:
                continue

            additional_ngrams = ngrams(doclike=noun_chunk,
                                       n=(1, allowed_pos_count),
                                       include_pos=self.textacy_pos_expand_nc)
            noun_chunk_ngram_subsets.update(set(additional_ngrams))

        return noun_chunk_ngram_subsets

    @staticmethod
    def chunk_intersect(source_chunks: Set[Span],
                        target_chunks: Set[Span],
                        normalize_fn: Callable) -> Set[Span]:
        target_surface_form = {normalize_fn(chunk) for chunk in target_chunks}
        intersected_chunks = {chunk for chunk in source_chunks
                              if normalize_fn(chunk) in target_surface_form}
        return intersected_chunks

    def _extract_shared_tokens(
        self, abstract: str, question: str, answers: str, nlp
    ) -> Sequence[QAIntersections]:

        abstract = nlp(abstract)

        lemmize_fn = ((lambda x: str(x.lemma_))
                      if self.use_lemma else
                      (lambda x: str(x)))
        stemmer = PorterStemmer()
        stem_fn = ((lambda x: stemmer.stem(lemmize_fn(x)))
                   if self.use_stem else lemmize_fn)
        chunk_intersect_fn = partial(self.chunk_intersect,
                                     normalize_fn=stem_fn)

        # process abstract and get chunks
        abstract_chunks = set(self.find_chunks(abstract))
        if self.expand_abstract:
            abstract_chunks = self.expand_chunks_set(abstract_chunks)

        # process question and get chunks
        question = nlp(question)
        question_chunks = set(self.find_chunks(question))
        if self.expand_question:
            question_chunks = self.expand_chunks_set(question_chunks)

        # intersect question and abstract
        question_chunks_in_abstract = chunk_intersect_fn(
            source_chunks=question_chunks,
            target_chunks=abstract_chunks
        )
        # we only keep the longest matches
        question_chunks_in_abstract_filtered = set(
            filter_spans(question_chunks_in_abstract)
        )
        # we get the matches from the abstract side
        abstract_chunks_in_question = chunk_intersect_fn(
            source_chunks=abstract_chunks,
            target_chunks=question_chunks_in_abstract_filtered
        )

        abstract_question_iscs = {
            match: MatchedChunk(source=abstract,
                                target=question,
                                match=match)
            for match in filter_spans(abstract_chunks_in_question)
        }

        for answer_span in answers['answer']:
            if answer_span['unanswerable']:
                continue

            for answer in answer_span['highlighted_evidence']:
                # process answer and get chunks
                answer = nlp(answer)
                answer_chunks = set(self.find_chunks(answer))
                if self.expand_answer:
                    answer_chunks = self.expand_chunks_set(answer_chunks)

                # intersect abstract, keep longest spans
                answer_chunks_in_abstract = set(filter_spans(
                    chunk_intersect_fn(source_chunks=answer_chunks,
                                       target_chunks=abstract_chunks)
                ))
                abstract_chunks_in_answer = chunk_intersect_fn(
                    source_chunks=abstract_chunks,
                    target_chunks=answer_chunks_in_abstract
                )
                abstract_answer_iscs = {
                    match: MatchedChunk(source=abstract,
                                        target=answer,
                                        match=match)
                    for match in filter_spans(abstract_chunks_in_answer)
                }

                # intersect same thing as before: we want the logest
                question_chunks_in_answer = chunk_intersect_fn(
                    source_chunks=question_chunks,
                    target_chunks=answer_chunks
                )
                question_chunks_in_abs_and_ans = chunk_intersect_fn(
                    source_chunks=question_chunks_in_abstract,
                    target_chunks=question_chunks_in_answer
                )
                question_chunks_in_abs_and_ans_filter = set(
                    filter_spans(question_chunks_in_abs_and_ans)
                )
                abstract_chunks_in_question_and_answer = filter_spans(
                    chunk_intersect_fn(
                        source_chunks=abstract_chunks,
                        target_chunks=question_chunks_in_abs_and_ans_filter
                    )
                )
                abstract_answer_question_iscs = {
                    match: MatchedChunk(source=abstract,
                                        target=question,
                                        match=match)
                    for match in abstract_chunks_in_question_and_answer
                }

                yield QAIntersections(
                    abstract_question=abstract_question_iscs,
                    abstract_answer=abstract_answer_iscs,
                    abstract_answer_question=abstract_answer_question_iscs
                )

    def transform(self, batch):
        nlp = self.get_spacy_pipeline()

        accumulator = defaultdict(list)

        for abstract, qas in zip(batch['abstract'], batch['qas']):
            questions, questions_answers = qas['question'], qas['answers']

            for question, answers in zip(questions, questions_answers):
                it = self._extract_shared_tokens(abstract=abstract,
                                                 question=question,
                                                 answers=answers,
                                                 nlp=nlp)
                question_abstract = 0
                answer_abstract = 0
                question_answer_abstract = 0

                for qa_intersections in it:
                    question_abstract += len(
                        qa_intersections.abstract_question
                    )
                    answer_abstract += len(
                        qa_intersections.abstract_answer
                    )
                    question_answer_abstract += len(
                        qa_intersections.abstract_answer_question
                    )

                accumulator['question_abstract'].append(
                    question_abstract
                )
                accumulator['answer_abstract'].append(
                    answer_abstract
                )
                accumulator['question_answer_abstract'].append(
                    question_answer_abstract
                )

        return accumulator


class MatchConfig(es.ConfigNode):
    dataset_name: es.ConfigParam(str) = 'qasper'
    split: es.ConfigParam(str)
    use_lemma: es.ConfigParam(bool) = False
    use_stem: es.ConfigParam(bool) = False
    spacy_model_name: es.ConfigParam(str) = 'en_core_web_sm'
    expand_abstract: es.ConfigParam(bool) = False
    expand_question: es.ConfigParam(bool) = False
    expand_answer: es.ConfigParam(bool) = False
    nprocs: es.ConfigParam(int) = None


@es.cli(MatchConfig)
def main(config: MatchConfig):
    mapper = Seq2SeqFeaturesMapperWithFocus(
        use_lemma=config.use_lemma,
        use_stem=config.use_stem,
        spacy_model_name=config.spacy_model_name,
        expand_abstract=config.expand_abstract,
        expand_question=config.expand_question,
        expand_answer=config.expand_answer
    )

    datasets.disable_progress_bar()

    dataset = datasets.load_dataset(path=config.dataset_name,
                                    split=config.split)

    dataset = dataset.map(mapper.transform,
                          batched=True,
                          num_proc=config.nprocs,
                          batch_size=None,
                          remove_columns=dataset.column_names)

    question_in_abstract = sum(1 if c > 0 else 0 for c in
                               dataset["question_abstract"])
    answer_in_abstract = sum(1 if c > 0 else 0 for c in
                             dataset["answer_abstract"])
    question_answer_in_abstract = sum(1 if c > 0 else 0 for c in
                                      dataset["question_answer_abstract"])

    print(f'question_in_abstract: {question_in_abstract/len(dataset):.2%}')
    print(f'answer_in_abstract: {answer_in_abstract/len(dataset):.2%}')
    print('question_answer_in_abstract: '
          f'{question_answer_in_abstract/len(dataset):.2%}')


if __name__ == '__main__':
    main()
