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


class BaseViz:
    def __init__(self, color_map: Dict[str, str], attribute: str):
        self.color_map = color_map
        self.attribute = attribute

    @classmethod
    def get_span_group_type(cls, span_group: SpanGroup) -> Union[str, None]:
        if span_group.type is not None:
            return span_group.type
        elif span_group.box_group is not None:
            return span_group.box_group.type
        else:
            return None

    @classmethod
    def draw_blocks(
        cls,
        image: PILImage,
        doc_spans: List[SpanGroup],
        pid: Optional[int] = None,
        color_map: Optional[Dict[str, str]] = None,
        token_boundary_width: int = 0,
        alpha: float = 0.25,
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
                type=cls.get_span_group_type(span),
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

    def __call__(self, doc: Document, path: Union[str, Path]):
        path = Path(path)

        if not(
            hasattr(doc, Pages) and
            hasattr(doc, TypedSentences) and
            hasattr(doc, Images)
        ):
            raise ValueError(f'Document must have `{Pages}`, `{Images}` and '
                             f'`{TypedSentences}` annotations!')

        pages: List[SpanGroup] = getattr(doc, 'pages', [])
        images: List[PILImage] = getattr(doc, 'images', [])

        for pid in range(len(pages)):
            viz = self.draw_blocks(
                image=images[pid],
                doc_spans=getattr(pages[pid], self.attribute, []),
                pid=pid,
                color_map=self.color_map,
                alpha=0.3
            )

            path.with_suffix("").mkdir(parents=True, exist_ok=True)
            viz.save(path.with_suffix("") / f"{pid}.png")


class TypedSentsViz(BaseViz):
    def __init__(self):
        from .typed_predictors import TypedBlockPredictor
        color_map = {
            TypedBlockPredictor.Title: 'red',
            TypedBlockPredictor.Text: 'blue',
            TypedBlockPredictor.Figure: 'green',
            TypedBlockPredictor.Table: 'yellow',
            TypedBlockPredictor.ListType: 'orange',
            TypedBlockPredictor.Other: 'grey',
            TypedBlockPredictor.RefApp: 'purple',
            TypedBlockPredictor.Abstract: 'magenta',
            TypedBlockPredictor.Preamble: 'cyan',
            TypedBlockPredictor.Caption: 'pink'
        }

        super().__init__(color_map=color_map, attribute='typed_sents')
