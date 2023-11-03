<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill Transformers Module

This repository contains the code supporting the Transformers models model for use with [Autodistill](https://github.com/autodistill/autodistill).

[Transformers](https://github.com/huggingface/transformers), maintained by Hugging Face, features a range of state of the art models for Natural Language Processing (NLP), computer vision, and more.

This package allows you to write a function that calls a Transformers object detection model and use it to automatically label data. You can use this data to train a fine-tuned model using an architecture supported by Autodistill (i.e. [YOLOv8](https://github.com/autodistil/autodistill-yolov8), [YOLOv5](https://github.com/autodistil/autodistill-yolov5), or [DETR](https://github.com/autodistil/autodistill-detr)).

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

## Installation

To use Transformers with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-transformers
```

## Quickstart

The following example shows how to use the Transformers module to label images using the [Owlv2ForObjectDetection](https://huggingface.co/google/owlv2-large-patch14-ensemble) model.

You can update the `inference()` functon to use any object detection model supported in the Transformers library.

```python
import cv2
import torch
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from autodistill_transformers import TransformersModel

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")


def inference(image, prompts):
    inputs = processor(text=prompts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]])

    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.1
    )[0]

    return results


base_model = TransformersModel(
    ontology=CaptionOntology(
        {
            "a photo of a person": "person",
            "a photo of a cat": "cat",
        }
    ),
    callback=inference,
)

# run inference
results = base_model.predict("image.jpg", confidence=0.1)

print(results)

# plot results
plot(
    image=cv2.imread("image.jpg"),
    detections=results,
    classes=base_model.ontology.classes(),
)

# label a directory of images
base_model.label("./context_images", extension=".jpeg")
```

## License

This project is licensed under an [MIT license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!