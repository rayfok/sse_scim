from dataclasses import dataclass
from typing import List, Sequence, Dict, Any, Optional

from pydantic import BaseModel, Field
import requests
from mmda.predictors.base_predictors.base_predictor import BasePredictor
from mmda.types.annotation import Annotation, SpanGroup
from mmda.types.document import Document

from .types import TypedSentences
from .typed_predictors import TypedBlockPredictor


class Instance(BaseModel):
    """Describes one Instance over which the model performs inference."""
    sentences: Sequence[str] = Field(
        description="A sequence of sentences to classify."
    )


class Instances(BaseModel):
    instances: List[Instance]


class FacetPrediction(BaseModel):
    label: str = Field(description="The predicted label.")
    confidence: float = Field(description="The confidence of the prediction.")


class Prediction(BaseModel):
    """Describes the outcome of inference for one Instance"""
    facets: List[List[FacetPrediction]] = Field(
        default_factory=list,
        description=("A dictionary of predicted facets, with the keys being "
                     "facet names and the values being the probabilities of "
                     "of each facet.")
    )


class Predictions(BaseModel):
    predictions: List[Prediction]


@dataclass
class SentencePredicted:
    sent: SpanGroup
    pred: List[FacetPrediction]


DEFAULT_ENDPOINT = 'http://sse.0-0-1.prod.models.s2.allenai.org/invocations'


class SkimmingPredictor(BasePredictor):
    REQUIRED_BACKENDS = None                        # type: ignore
    REQUIRED_DOCUMENT_FIELDS = [TypedSentences]     # type: ignore

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        headers: Optional[Dict[str, Any]] = None
    ) -> None:
        self.endpoint = endpoint
        self.headers = headers or {'accept': 'application/json',
                                   'Content-Type': 'application/json'}
        super().__init__()

    def predict(self, doc: Document) -> List[SentencePredicted]:
        super().predict(doc)

        predict_on = [
            sent for sent in doc.typed_sents    # type: ignore
            if sent.type == TypedBlockPredictor.ListType or
            sent.type == TypedBlockPredictor.Text
        ]

        data = Instances(instances=[
            Instance(sentences=[str(t.text) for t in predict_on])
        ])

        response = requests.post(
            url=self.endpoint,
            json=data.dict(),
            headers=self.headers
        )
        predictions = Predictions(**response.json())
        return [
            SentencePredicted(sent=sent, pred=pred)
            for sent, pred in zip(predict_on, predictions.predictions[0].facets)
        ]
