from collections import defaultdict

import spacy
from spacy.symbols import VERB

nlp = spacy.load("en_core_web_sm")

def classify_novelty(ref_sents):
    novelty_verbs = {"categorize", "categorise" "formalize", "hypothesize", "propose", "posit", "contribute", "develop", "benchmark", "augment", "demonstrate", "explore", "extend", "emphasize", "solve", "investigate", "summarize", "uncover", "rethink", "speculate", "motivate"}
    novelty_nouns = {"new", "novel", "differ"}
    novelty_predictions = defaultdict(lambda: False)
    for sent in ref_sents:
        page = sent.box_group.boxes[0].page
        # if page > 2:
        #     continue
        tokens = [t.lower() for t in clean_sentence(sent.text).split()]
        if any(kw in tokens for kw in ["we", "us", "our"]):
            doc = nlp(sent.text)
            for token in doc:
                if (token.pos_ == VERB and token.lemma_ in novelty_verbs) or token.lemma_ in novelty_nouns:
                    novelty_predictions[sent.id] = True
                    break
    return novelty_predictions


def clean_sentence(sentence):
    cleaned = sentence.replace("\ufb01", "fi")
    cleaned = cleaned.replace("\ufb02", "fl")
    cleaned = cleaned.replace("\u2019", "'")
    cleaned = cleaned.replace("\u00ad", "-")
    cleaned = cleaned.encode("ascii", "ignore")
    cleaned = cleaned.decode()
    cleaned = cleaned.replace("- ", "")
    return cleaned
