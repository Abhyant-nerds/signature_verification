import numpy as np

from signature_verification.preprocessing import PreprocessConfig, preprocess_signature


def test_preprocessing_shape_and_range():
    image = np.full((100, 200), 255, dtype=np.uint8)
    image[40:60, 30:170] = 0
    tensor, quality, processed = preprocess_signature(image, PreprocessConfig(blur_threshold=0))
    assert tensor.shape == (3, 224, 224)
    assert processed.shape == (224, 224)
    assert tensor.min() >= -1 and tensor.max() <= 1
    assert quality.usable


def test_blank_image_is_unusable():
    image = np.full((100, 200), 255, dtype=np.uint8)
    _, quality, _ = preprocess_signature(image)
    assert not quality.usable
    assert "BLANK" in quality.reasons
