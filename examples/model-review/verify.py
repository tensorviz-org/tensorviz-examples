"""Check the example's shape contract and numerical review boundary on CPU."""

import torch

from after import ReviewNet as After
from before import ReviewNet as Before


def set_demo_weights(model):
    """Use paired positive/negative features and select the first pair at output."""
    with torch.no_grad():
        model.hidden.weight.copy_(torch.cat((torch.eye(4), -torch.eye(4))))
        model.output.weight.zero_()
        model.output.weight[0, 0] = 1.0
        model.output.weight[1, 4] = 1.0


def run_and_record(model, x):
    shapes = {}
    hooks = []
    for name, layer in model.named_children():
        def record(_layer, _args, result, name=name):
            shapes[name] = tuple(result.shape)
        hooks.append(layer.register_forward_hook(record))
    try:
        with torch.inference_mode():
            output = model(x)
    finally:
        for hook in hooks:
            hook.remove()
    return output, shapes


def main():
    before = Before().cpu().eval()
    after = After().cpu().eval()
    assert isinstance(before.activation, torch.nn.ReLU)
    assert isinstance(after.activation, torch.nn.GELU)
    assert isinstance(after.norm, torch.nn.LayerNorm) and after.norm.normalized_shape == (8,)
    set_demo_weights(before)
    set_demo_weights(after)
    torch.testing.assert_close(before.hidden.weight, after.hidden.weight)
    torch.testing.assert_close(before.output.weight, after.output.weight)

    for batch in (1, 3, 8):
        x = torch.tensor([[-1.0, 0.5, 2.0, -2.0]]).repeat(batch, 1)
        before_result, before_shapes = run_and_record(before, x)
        after_result, after_shapes = run_and_record(after, x)
        assert before_shapes == {
            "hidden": (batch, 8), "activation": (batch, 8), "output": (batch, 2)
        }, before_shapes
        assert after_shapes == {
            "hidden": (batch, 8), "activation": (batch, 8),
            "norm": (batch, 8), "output": (batch, 2)
        }, after_shapes
        assert before_result.shape == after_result.shape == (batch, 2)
        assert torch.isfinite(before_result).all() and torch.isfinite(after_result).all()
        torch.testing.assert_close(before_result, torch.tensor([[0.0, 1.0]]).repeat(batch, 1))
        assert not torch.allclose(before_result, after_result), (
            "This intentional architecture edit should change the demo's numerical result."
        )
        print(f"PASS batch={batch}: both models return [{batch}, 2]; numerical results differ")

    for name, model in (("before", before), ("after", after)):
        try:
            with torch.inference_mode():
                model(torch.zeros(1, 5))
        except RuntimeError as error:
            assert "mat1 and mat2 shapes cannot be multiplied" in str(error), str(error)
            print(f"PASS {name}: incompatible input [1, 5] is rejected")
        else:
            raise AssertionError(f"{name} unexpectedly accepted input [1, 5]")

    print("All CPU checks passed. Shapes are a contract; they are not proof of equivalence.")


if __name__ == "__main__":
    main()
