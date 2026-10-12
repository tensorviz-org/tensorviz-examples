# Fix “mat1 and mat2 shapes cannot be multiplied” in PyTorch

Your convolution runs, flattening runs, and then `nn.Linear` fails:

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x512 and 128x5)
```

The useful question is **how many features reach the linear layer for each
image?** Here it receives 512, but was constructed to accept 128. The batch size
of 2 is not the problem.

This small, original CPU example reproduces that mistake, fixes it for the
intended image size, and checks what changes when the image size changes. No
dataset, pretrained weights, GPU or tensorViz installation is needed.

## Run the example

The exact tested stack is **Python 3.12.14 and PyTorch 2.7.1**. Use an existing
compatible environment if you have one; do not replace your project's PyTorch
or CUDA installation for this exercise. From this folder, run:

```bash
python verify.py
```

If you need a separate environment on macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py
```

That optional installation downloads PyTorch and its dependencies, not a model.
The pin was checked with Python 3.12; it does not establish compatibility with
every Python version or operating system.

`verify.py` catches the intentional errors and checks both working models. Its
last line should be:

```text
All CPU checks passed. Fix the feature contract; do not reshape away the mismatch.
```

## Reproduce the failure

[`broken.py`](broken.py) contains this model:

```python
from torch import nn


class BrokenClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(128, 5)  # Wrong feature count.

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)
```

Run the following from this folder to see the uncaught error:

```python
import torch
from broken import BrokenClassifier

model = BrokenClassifier().eval()
model(torch.zeros(2, 3, 16, 16))
```

The tensor contains two RGB images, each 16 pixels high and 16 pixels wide.
Follow the dimensions before changing code:

| Boundary | Shape | Why |
| --- | --- | --- |
| Input | `[2, 3, 16, 16]` | Batch, channels, height, width |
| Convolution | `[2, 8, 16, 16]` | Eight output channels; padding preserves spatial size |
| ReLU | `[2, 8, 16, 16]` | Elementwise activation preserves shape |
| Max pooling | `[2, 8, 8, 8]` | A 2-by-2 pool halves both spatial dimensions here |
| Flatten | `[2, 512]` | `8 * 8 * 8 = 512` values per image |
| Broken linear layer | Error | Expects 128 values per image |

`nn.Linear(128, 5)` stores a weight of shape `[5, 128]`. The operation multiplies
the `[2, 512]` input by the transposed `[128, 5]` weight. The inner dimensions,
512 and 128, must match. Changing the batch size cannot repair them.

## Fix the head for the intended input

If 16-by-16 images are the intended input, change the head to:

```python
self.classifier = nn.Linear(8 * 8 * 8, 5)
```

The complete corrected model is [`fixed.py`](fixed.py). Keep
`nn.Flatten(start_dim=1)` so each image remains a separate row. The flow is now
`[batch, 3, 16, 16] → [batch, 512] → [batch, 5]`.

The check covers batch sizes 1, 2 and 5. It also compares a full batch with the
same images run one at a time, so the expected row count is not the only check
of the batch behavior. These are untrained example models; passing the check
does not establish useful predictions or training quality.

Changing `in_features` changes the head's parameter shape. Existing weights for
a 128-wide head cannot be loaded unchanged into this 512-wide head. In a real
project, first establish whether the input pipeline, feature extractor or head
is wrong before deciding which one to change.

## Why reshaping away the error is dangerous

This can make the multiplication run:

```python
flat = torch.zeros(2, 512)
wrong_batch = flat.reshape(-1, 128)
print(wrong_batch.shape)  # torch.Size([8, 128])
```

It has turned two images into eight rows. The broken head now returns eight
predictions for two images. It has not learned to accept 512 features per image;
the code has changed what a row means. `verify.py` checks this counterexample
using the actual convolution features.

Reshape changes how existing elements are grouped. It does not infer the
intended sample boundaries or repair a semantically wrong feature count.

## What if the image size changes?

For `[2, 3, 16, 20]`, the same feature extractor produces `[2, 8, 8, 10]`.
Flattening gives 640 values per image, so the corrected 512-wide head fails:

```text
mat1 and mat2 shapes cannot be multiplied (2x640 and 512x5)
```

The check also covers `[2, 3, 20, 16]`, which produces the same feature count.
The correction above establishes the intended 16-by-16 input contract; it does
not make the model accept arbitrary image sizes. Other input sizes can happen
to produce 512 flattened values without being semantically appropriate.

If the task should support varying spatial dimensions, one architecture choice
is global average pooling. [`pooled.py`](pooled.py) adds these layers after the
feature extractor:

```python
self.pool = nn.AdaptiveAvgPool2d((1, 1))
self.flatten = nn.Flatten(start_dim=1)
self.classifier = nn.Linear(8, 5)
```

The forward pass applies `features`, then `pool`, then `flatten`, then
`classifier`. Each channel is averaged to one value, giving
`[batch, 8, height, width] → [batch, 8, 1, 1] → [batch, 8] → [batch, 5]`.
The included checks cover the square input and both non-square inputs above.

This is a **different architecture**. Averaging discards spatial detail in the
feature map. Its head has different weights and computation; it is not
numerically equivalent to the fixed-resolution model or a drop-in fix for a
pretrained checkpoint. Choose it only when that modeling tradeoff fits the task.

## Optional: inspect the corrected model in tensorViz

In desktop VS Code, install tensorViz from the
[VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=tensorviz.tensorviz-vscode)
or [Open VSX](https://open-vsx.org/extension/tensorviz/tensorviz-vscode),
open `fixed.py`, and choose **Open as Graph** above `FixedClassifier`. Select
the Python environment used for the check. For a captured run, use a CPU
float32 input of shape `[2, 3, 16, 16]` in evaluation mode. First use may require
the extension's separate service setup; see the [setup guide](https://tensorviz.dev/install.html#first-graph).

After a successful capture, inspect the shapes entering and leaving the flatten
operation and the linear layer, and use source navigation to find their
definitions. A source-only preview is not a measurement of a forward pass.
Trying the broken model does not establish that a failed capture will return a
complete graph or automatically diagnose the cause.

The shapes in this guide were verified with the standalone CPU script. A
tensorViz capture and rendered UI check have **not** been run for this example.
See [validation notes](../../docs/validation.md#shape-mismatch-exercise) for the
tested scope and [support instructions](../../docs/support.md) if you get stuck.

The original example code and this guide are covered by the repository's
[MIT license](../../LICENSE).
