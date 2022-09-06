import numpy as np

from weak_label.labeling_utils import export_labeled_dataset, load_dataset
from weak_label.lfs import Annotator


def main():
    dataset = load_dataset(src="output/dataset/raw/")
    dataset = dataset[:100]

    ann = Annotator()
    lfs = [
        ann.contains_author_statement,
        ann.exceeds_min_length,
        ann.exceeds_max_length,
        # ann.contains_method_statement,
        ann.contains_result_statement,
        ann.contains_objective_statement,
        ann.contains_novelty_statement,
        ann.is_in_acknowledgements,
        ann.contains_link,
        ann.contains_arxiv,
        ann.is_list_elem_in_introduction,
    ]

    labels = ann.apply_labeling_functions(dataset, lfs)
    lf_summary = ann.get_lf_summary(labels, lfs)
    print(lf_summary)

    preds = ann.predict(labels, agg_model="label")
    for pred, x in zip(preds, dataset):
        label, score = np.argmax(pred), pred[np.argmax(pred)]
        x.label = int(label)
        x.score = float(score)

    export_labeled_dataset(dataset, dst="output/snorkel/weak_labels.json")


if __name__ == "__main__":
    main()
