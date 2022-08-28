import re
from collections import defaultdict

import spacy
from pdf2sents.typed_predictors import TypedBlockPredictor
from spacy.symbols import VERB

nlp = spacy.load("en_core_web_sm")

NOVELTY_KWS = {
    "categorize",
    "categorise",
    "formalize",
    "hypothesize",
    "propose",
    "posit",
    "develop",
    "benchmark",
    "augment",
    "demonstrate",
    "explore",
    "extend",
    "emphasize",
    "solve",
    "investigate",
    "summarize",
    "uncover",
    "rethink",
    "speculate",
    "motivate",
    "extend",
    "differ",
    "contrast",
    "new",
    "novel",
}

CONTRIBUTION_KWS = {"contribution", "contributions", "contribute"}

METHOD_KWS = {}

RESULT_KWS = {
    "find",
    "show",
    "reveal",
    "highlight",
    "achieve",
    "obtain",
    "exceed",
    "outperform",
    "surpass",
}

OBJECTIVE_KWS = {"aim", "goal", "purpose", "motivation", "objective"}


def classify_novelty(ref_sents):
    novelty_predictions = set()
    for sent in ref_sents:
        page = sent.box_group.boxes[0].page
        doc = nlp(sent.text)
        for token in doc:
            if (
                token.lemma_ in NOVELTY_KWS
                and sentence_has_author_intent(sent)
                and is_sentence_in_section(
                    sent,
                    [
                        "abstract",
                        "introduction",
                        "related work",
                        "recent work",
                        "conclusion",
                    ],
                )
            ):
                novelty_predictions.add(sent.id)
                break
            if token.lemma_ in CONTRIBUTION_KWS and page <= 2:
                novelty_predictions.add(sent.id)
                break
    return novelty_predictions


def classify_method(ref_sents):
    method_predictions = set()
    for sent in ref_sents:
        doc = nlp(sent.text)
        if sentence_has_author_intent(sent) and is_sentence_in_section(
            sent, ["abstract", "introduction", "method", "approach"]
        ):
            for token in doc:
                if token.lemma_ in METHOD_KWS:
                    method_predictions.add(sent.id)
                    break
    return method_predictions


def classify_result(ref_sents):
    result_predictions = set()
    for sent in ref_sents:
        doc = nlp(sent.text)
        if (
            sentence_has_author_intent(sent)
            and is_sentence_in_section(sent, ["abstract", "introduction", "conclusion"])
        ) or is_sentence_in_section(sent, ["result", "experiment"]):
            for token in doc:
                if token.lemma_ in RESULT_KWS:
                    result_predictions.add(sent.id)
                    break
    return result_predictions


def classify_objective(ref_sents):
    objective_predictions = set()
    for sent in ref_sents:
        doc = nlp(sent.text)
        if sentence_has_author_intent(sent) and is_sentence_in_section(
            sent, ["abstract", "introduction", "conclusion"]
        ):
            for token in doc:
                if token.text in OBJECTIVE_KWS:
                    objective_predictions.add(sent.id)
                    break
    return objective_predictions


def is_sentence_in_section(sentence, sections):
    return any(section in sentence.section.lower() for section in sections)


def assign_sections_to_sentences(sents):
    sent_sect_map = {}
    sent_fields = [
        TypedBlockPredictor.Text,
        TypedBlockPredictor.ListType,
        TypedBlockPredictor.Abstract,
    ]
    sect_fields = [TypedBlockPredictor.Title]
    sents = [s for s in sents if s.type in sent_fields + sect_fields]
    sents = sorted(sents, key=lambda x: x.spans[0].start)

    all_sections = {}  # { <section number>: <section text> }
    current_section = ""
    for x in sents:
        if x.type in sent_fields:
            sent_sect_map[x.text] = current_section
        else:
            current_section = x.text
            match = re.search(r"[-+]?\d*\.\d+|\d+", current_section)
            if match:
                sect_number = match.group()
                all_sections[sect_number] = current_section

    for sent, sect in sent_sect_map.items():
        match = re.search(r"[-+]?\d*\.\d+|\d+", sect)
        if match:
            sect_number = match.group()
            sect_number_parts = sect_number.split(".")

            if len(sect_number_parts) == 3:
                first_level_header = all_sections.get(sect_number_parts[0], "")
                second_level_header = all_sections.get(
                    ".".join(sect_number_parts[:2]), ""
                )
                sent_sect_map[
                    sent
                ] = f"{first_level_header} @@ {second_level_header} @@ {sect}"
            elif len(sect_number_parts) == 2:
                first_level_header = all_sections.get(sect_number_parts[0], "")
                sent_sect_map[sent] = f"{first_level_header} @@ {sect}"
    return sent_sect_map


def clean_sentence(sentence):
    cleaned = sentence.replace("\ufb01", "fi")
    cleaned = cleaned.replace("\ufb02", "fl")
    cleaned = cleaned.replace("\u2019", "'")
    cleaned = cleaned.replace("\u00ad", "-")
    cleaned = cleaned.encode("ascii", "ignore")
    cleaned = cleaned.decode()
    cleaned = cleaned.replace("- ", "")
    return cleaned


def sentence_has_author_intent(sentence):
    author_intent_tokens = ["we", "us", "our"]
    tokens = clean_sentence(sentence.text).lower().split()
    return any(t in tokens for t in author_intent_tokens)


def extract_verbs(sentence):
    verbs = []
    doc = nlp(sentence.text)
    for token in doc:
        if token.pos == VERB:
            verbs.append(token.lemma_)
    return verbs
