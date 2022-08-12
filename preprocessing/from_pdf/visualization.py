from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from mmda.types.annotation import SpanGroup

import layoutparser as lp

from mmda.types.document import Document
from mmda.types.names import Images, Pages

from .types import TypedSentences


def draw_blocks(
    image,
    doc_spans: List[SpanGroup],
    pid=None,
    color_map=None,
    token_boundary_width=0,
    alpha=0.25,
    **kwargs,
):

    w, h = image.size
    layout = [
        lp.TextBlock(
            lp.Rectangle(
                *box
                .get_absolute(page_height=h, page_width=w)
                .coordinates
            ),
            type=(span.type or span.box_group.type),
            text=(doc_spans[0].text or ' '.join(span.symbols)),
        )
        for span in doc_spans
        for box in span.box_group.boxes
        if (box.page == pid if pid is not None else True)
    ]

    return lp.draw_box(
        image,
        layout,
        color_map=color_map,
        box_color='grey' if not color_map else None,
        box_width=token_boundary_width,
        box_alpha=alpha,
        **kwargs,
    )


def visualize_typed_sentences(
    doc: Document,
    path: Union[str, Path],
    attr: str = 'typed_sents',
    color_map: Optional[Dict[str, str]] = None
):
    from .typed_predictors import TypedBlockPredictor

    path = Path(path)
    # label_fn = label_fn or (label_fn x: x.type)

    if color_map is None:
        color_map = {TypedBlockPredictor.Title: 'red',
                     TypedBlockPredictor.Text: 'blue',
                     TypedBlockPredictor.Figure: 'green',
                     TypedBlockPredictor.Table: 'yellow',
                     TypedBlockPredictor.ListType: 'orange',
                     TypedBlockPredictor.Other: 'grey',
                     TypedBlockPredictor.RefApp: 'purple',
                     TypedBlockPredictor.Abstract: 'magenta',
                     TypedBlockPredictor.Preamble: 'cyan',
                     TypedBlockPredictor.Caption: 'pink'}

    if not(hasattr(doc, Pages) and
           hasattr(doc, TypedSentences) and
           hasattr(doc, Images)):
        raise ValueError(f'Document must have `{Pages}`, `{Images}` and '
                         f'`{TypedSentences}` annotations!')

    for pid in range(len(doc.pages)):
        viz = draw_blocks(doc.images[pid],
                          getattr(doc.pages[pid], attr),
                          pid=pid,
                          color_map=color_map,
                          alpha=0.3)

        path.with_suffix("").mkdir(parents=True, exist_ok=True)
        viz.save(path.with_suffix("") / f"{pid}.png")
