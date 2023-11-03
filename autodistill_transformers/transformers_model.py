import os
from dataclasses import dataclass

import torch

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from PIL import Image

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TransformersModel(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology, callback):
        self.callback = callback
        self.ontology = ontology

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        image = Image.open(input)

        with torch.no_grad():
            outputs = self.callback(image, [self.ontology.prompts()])

        detections = sv.Detections.from_transformers(outputs)

        detections = detections[detections.confidence > confidence]

        return detections
