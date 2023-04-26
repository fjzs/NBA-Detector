import albumentations as A
from nba_detector.transformations import get_transformation
import pytest

def test_get_transformation_bad_format():
    bad_input_examples = ["", "COCO", "Coco", "124", "Yolo", "pascal voc"]
    for bad_input in bad_input_examples:
        with pytest.raises(ValueError):
            get_transformation(bad_input)

def test_get_transformation_return_type():
    output = get_transformation()
    assert(type(output) == A.Compose)
