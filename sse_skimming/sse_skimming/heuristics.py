import re
from collections import defaultdict

import spacy
from pdf2sents.typed_predictors import TypedBlockPredictor
from spacy.symbols import VERB

nlp = spacy.load("en_core_web_sm")

def classify_novelty(ref_sents):
    novelty_verbs = {"categorize", "categorise" "formalize", "hypothesize", "propose", "posit", "contribute", "develop", "benchmark", "augment", "demonstrate", "explore", "extend", "emphasize", "solve", "investigate", "summarize", "uncover", "rethink", "speculate", "motivate", "extend"}
    novelty_nouns = {"new", "novel", "differ", "contrast"}
    contribution_nouns = {"contribution", "contributions"}
    novelty_predictions = defaultdict(lambda: False)
    for sent in ref_sents:
        page = sent.box_group.boxes[0].page
        tokens = [t.lower() for t in clean_sentence(sent.text).split()]
        doc = nlp(sent.text)
        for token in doc:
            if any(kw in tokens for kw in ["we", "us", "our"]):
                if (token.pos_ == VERB and token.lemma_ in novelty_verbs) or token.lemma_ in novelty_nouns:
                    novelty_predictions[sent.id] = True
                    break

            if token.lemma_ in contribution_nouns and page <= 2:
                novelty_predictions[sent.id] = True
                break
    return novelty_predictions


def assign_sections_to_sentences(sents):
    sent_sect_map = {}
    sent_fields = [TypedBlockPredictor.Text,
        TypedBlockPredictor.ListType,
        TypedBlockPredictor.Abstract]
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
