from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL.Image import Image as PILImage

from mmda.types.annotation import SpanGroup

from layoutparser.elements.layout_elements import (
    TextBlock,
    Rectangle
)
from layoutparser.visualization import draw_box

from mmda.types.document import Document
from mmda.types.names import Images, Pages

from .types import TypedSentences


def get_span_group_type(span_group: SpanGroup) -> Union[str, None]:
    if span_group.type is not None:
        return span_group.type
    elif span_group.box_group is not None:
        return span_group.box_group.type
    else:
        return None


def draw_blocks(
    image: PILImage,
    doc_spans: List[SpanGroup],
    pid=None,
    color_map=None,
    token_boundary_width=0,
    alpha=0.25,
    **kwargs,
):

    w, h = image.size
    layout = [
        TextBlock(
            Rectangle(
                *box
                .get_absolute(page_height=h, page_width=w)
                .coordinates
            ),
            type=get_span_group_type(span),
            text=(doc_spans[0].text or ' '.join(span.symbols)),
        )
        for span in doc_spans
        for box in getattr(span.box_group, 'boxes', [])
        if (box.page == pid if pid is not None else True)
    ]

    return draw_box(
        image,
        layout,      # type: ignore
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

    pages: List[SpanGroup] = getattr(doc, 'pages', [])
    images: List[PILImage] = getattr(doc, 'images', [])

    for pid in range(len(pages)):
        viz = draw_blocks(images[pid],
                          getattr(pages[pid], attr),
                          pid=pid,
                          color_map=color_map,
                          alpha=0.3)

        path.with_suffix("").mkdir(parents=True, exist_ok=True)
        viz.save(path.with_suffix("") / f"{pid}.png")
