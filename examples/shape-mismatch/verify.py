"""Reproduce the error and check two explicitly different repairs on CPU."""

import torch

from broken import BrokenClassifier
from fixed import FixedClassifier
from pooled import PooledClassifier


def image_batch(batch, height=16, width=16):
    """Distinct, deterministic synthetic pixels; no dataset or model download."""
    values = torch.arange(batch * 3 * height * width, dtype=torch.float32)
    return values.reshape(batch, 3, height, width) / (3 * height * width)


def expect_linear_error(model, x, expected_dimensions):
    try:
        with torch.inference_mode():
            model(x)
    except RuntimeError as error:
        message = str(error)
        assert "mat1 and mat2 shapes cannot be multiplied" in message, message
        assert expected_dimensions in message, message
        return message
    raise AssertionError("The mismatched Linear input unexpectedly succeeded.")


def run_and_record(model, x):
    shapes = {}
    hooks = []
    for name, layer in model.named_children():
        def record(_layer, _args, result, name=name):
            shapes[name] = tuple(result.shape)
        hooks.append(layer.register_forward_hook(record))
    try:
        with torch.inference_mode():
            result = model(x)
    finally:
        for hook in hooks:
            hook.remove()
    return result, shapes


def main():
    torch.manual_seed(0)
    broken = BrokenClassifier().cpu().eval()
    fixed = FixedClassifier().cpu().eval()
    pooled = PooledClassifier().cpu().eval()

    for batch in (1, 2, 5):
        x = image_batch(batch)
        with torch.inference_mode():
            feature_map = broken.features(x)
            flattened = broken.flatten(feature_map)
        assert feature_map.shape == (batch, 8, 8, 8), feature_map.shape
        assert flattened.shape == (batch, 512), flattened.shape
        message = expect_linear_error(broken, x, f"({batch}x512 and 128x5)")
        print(f"EXPECTED ERROR batch={batch}: {message}")

        result, shapes = run_and_record(fixed, x)
        assert shapes == {
            "features": (batch, 8, 8, 8), "flatten": (batch, 512),
            "classifier": (batch, 5),
        }, shapes
        assert result.shape == (batch, 5) and torch.isfinite(result).all()
        with torch.inference_mode():
            one_at_a_time = torch.cat([fixed(sample.unsqueeze(0)) for sample in x])
        torch.testing.assert_close(result, one_at_a_time, rtol=1e-5, atol=1e-5)
        print(f"PASS fixed: [{batch}, 3, 16, 16] -> [{batch}, 512] -> [{batch}, 5]")

    for height, width in ((16, 20), (20, 16)):
        x = image_batch(2, height, width)
        with torch.inference_mode():
            flattened = fixed.flatten(fixed.features(x))
        assert flattened.shape == (2, 640)
        message = expect_linear_error(fixed, x, "(2x640 and 512x5)")
        print(f"EXPECTED fixed-resolution limit [2, 3, {height}, {width}]: {message}")

    for batch, height, width in ((1, 16, 16), (2, 16, 16), (5, 16, 16),
                                 (2, 16, 20), (3, 20, 16)):
        result, shapes = run_and_record(pooled, image_batch(batch, height, width))
        assert shapes == {
            "features": (batch, 8, height // 2, width // 2),
            "pool": (batch, 8, 1, 1), "flatten": (batch, 8),
            "classifier": (batch, 5),
        }, shapes
        assert result.shape == (batch, 5) and torch.isfinite(result).all()
        print(f"PASS pooled: [{batch}, 3, {height}, {width}] -> [{batch}, 8] -> [{batch}, 5]")

    with torch.inference_mode():
        flat = broken.flatten(broken.features(image_batch(2)))
        wrong_batch = flat.reshape(-1, 128)
        wrong_output = broken.classifier(wrong_batch)
    assert flat.shape == (2, 512)
    assert wrong_batch.shape == (8, 128) and wrong_output.shape == (8, 5)
    assert wrong_output.shape[0] != 2, "A reshape must not silently redefine the batch."
    print("PASS reshape counterexample: [2, 512] became [8, 128], giving eight outputs for two images")
    print("All CPU checks passed. Fix the feature contract; do not reshape away the mismatch.")


if __name__ == "__main__":
    main()
