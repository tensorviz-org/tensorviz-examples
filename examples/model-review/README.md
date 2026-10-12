# Same interface, different computation

You are reviewing a small model edit: **ReLU becomes GELU, and LayerNorm is added**.
The input and output dimensions do not change. What can you learn from the graph,
and what still needs an executable check?

This is a CPU exercise with no pretrained weights or external dataset. It uses
original toy models so you can share the files without sharing your own project.

## Setup

The exact tested example stack is **Python 3.12.14 and PyTorch 2.7.1**. The
extension has a separate [compatibility matrix](../../README.md#compatibility).
Use an existing compatible project environment if you have one; do not replace
its PyTorch or CUDA installation for this exercise.

This folder includes an optional VS Code recommendation for tensorViz. It does
not install the extension or other packages automatically.

For a separate environment on macOS or Linux, from this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py
```

The installation command downloads PyTorch and its dependencies. It does not
download a model. Creating an environment is optional if your selected Python
already has PyTorch. The pinned example setup is not a claim that Python 3.14
works with this particular older wheel; use Python 3.12 for this pin.

PowerShell uses `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1` instead.
The example's Windows execution and tensorViz's Windows UI were not verified in
this check; use the [published platform limits](https://tensorviz.dev/install.html#compatibility)
when choosing your setup.

Successful output ends with:

```text
All CPU checks passed. Shapes are a contract; they are not proof of equivalence.
```

## Explore the original model

1. Open **this `model-review` folder** in desktop VS Code 1.106+ and install
   tensorViz from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=tensorviz.tensorviz-vscode)
   or [Open VSX](https://open-vsx.org/extension/tensorviz/tensorviz-vscode).
2. Open `before.py` and choose **Open as Graph** above `ReviewNet`.
3. Choose the same Python environment that ran `verify.py`. The saved
   `tensorviz.toml` provides the `before` profile: CPU, evaluation mode, and a
   float32 input of shape `[1, 4]`. If asked for inputs, use those values.
4. Follow the hidden linear layer, ReLU, and output linear layer. Their output
   shapes are `[1, 8]`, `[1, 8]`, and `[1, 2]`. Select a layer and follow its source
   link back to the model definition.

If tensorViz needs its separate service dependencies, review its setup prompt.
First-run downloads are not included in a fixed-time promise. The manual model
workflow does not require a coding agent, API key or usage-analytics sharing.

## Review the edit

Open `after.py` and choose **Open as Graph** above its `ReviewNet`. Use the `after`
profile or the same `[1, 4]` input. Find GELU and the new LayerNorm between the
hidden layer and output layer. Both preserve the hidden width of 8.

| Boundary | Before | After |
| --- | --- | --- |
| Input | `[batch, 4]` | `[batch, 4]` |
| Hidden linear output | `[batch, 8]` | `[batch, 8]` |
| Activation | ReLU, `[batch, 8]` | GELU, `[batch, 8]` |
| Normalization | None | LayerNorm, `[batch, 8]` |
| Final output | `[batch, 2]` | `[batch, 2]` |

To read the source difference in a terminal with Git installed:

```bash
git diff --no-index -- before.py after.py
```

Git exits with status 1 here because the files differ; that is expected. You can
also use VS Code's **Select for Compare** / **Compare with Selected** file actions.
These two static example files do not create a pull request or two Git revisions.
tensorViz's Git/PR architecture-review workflow is a separate product workflow.

## Check the numerical boundary

`verify.py` gives both models the same controlled linear weights. For the input
`[-1, 0.5, 2, -2]`, the original network returns `[0, 1]`. The edited network
returns two finite values with the same shape, but they are not the same values.

The check repeats this for batch sizes 1, 3 and 8 and records every module's output
shape. It also verifies that `[1, 5]` is rejected because the first linear layer
expects a last dimension of 4. This deliberate failure is caught by the check; it
is not an automatic-error-detection claim about tensorViz.

The graph profiles use seeded random initialization and input. They show the
same structure and shape contract, but **do not reproduce the controlled values
from `verify.py`**. The controlled-weight numerical demonstration runs in the
standalone check.

The lesson is limited and useful: a shape-preserving edit can change computation.
The graph makes the changed operations inspectable. It does not tell you whether
the edit improves training, matches an intended algorithm or preserves behavior
on inputs you have not tested.

## What to try next

- Change the saved batch size and confirm the module-boundary shapes again.
- Replace `nn.LayerNorm(8)` with `nn.Identity()` in `after.py`, rerun `verify.py`,
  and inspect why the normalization check now fails. Update an expectation only
  after deciding what behavior the revised exercise is supposed to teach.
- Bring a small model you are permitted to share and write its expected input and
  output contract before capturing it.

If setup fails, use the [support guide](../../docs/support.md). For what was
actually tested, see [validation notes](../../docs/validation.md).
