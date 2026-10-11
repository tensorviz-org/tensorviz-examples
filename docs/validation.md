# Validation notes

The examples are designed to be checked locally without the extension. The
command is:

```bash
cd examples/model-review
python verify.py
```

The check must pass on CPU before an example change is proposed. It covers:

- input and module-boundary shapes for batch sizes 1, 3 and 8;
- the intended ReLU-to-GELU change and the added LayerNorm;
- output finiteness and the unchanged `[batch, 2]` interface;
- shared, controlled linear weights across both models;
- a known ReLU result and a numerical difference after the architecture edit;
- rejection of a deliberately incompatible last input dimension.

These are checks of the educational example. They do not demonstrate training
quality, equivalence for all inputs, extension compatibility on every platform,
or successful capture in the installed tensorViz UI.

## October 11, 2026 local check

The initial exercise was tested with Python **3.12.14** and PyTorch **2.7.1** on
macOS Apple Silicon, using an existing environment and CPU tensors. No packages
were installed or changed. `requirements.txt` pins the PyTorch version used for
that example check; it is not a requirement to downgrade an existing project.
All checks passed. An in-memory mutation replacing the edited model's GELU with
ReLU made the check fail as intended; no modified example was saved. Python,
JSON, YAML and TOML files parsed, profile targets were consistent, and local
Markdown link targets existed.

## Saved-profile backend capture

On October 11, 2026, both unchanged `tensorviz.toml` profiles were also checked
through the backend from tensorViz **0.1.11's released source**, using the same
existing Python 3.12.14 / PyTorch 2.7.1 environment. This used the product's saved
profile resolution, resolved execution plan, capture worker and graph projection.
It did not substitute a direct model call for the capture check.

Both profiles resolved successfully, used one configured input candidate and
returned a complete capture with a current execution plan. The captured input was
CPU float32 `[1, 4]`; the output was `[1, 2]` in both cases.

| Captured boundary | Before | After |
| --- | --- | --- |
| Hidden linear output | `[1, 8]` | `[1, 8]` |
| Activation output | ReLU, `[1, 8]` | GELU, `[1, 8]` |
| Normalization output | None | LayerNorm, `[1, 8]` |
| Final output | `[1, 2]` | `[1, 2]` |

The operation records include `aten.relu` in the original and `aten.gelu` plus
`aten.native_layer_norm` in the edited model. All child module invocations have
source bindings. Both captures report no coverage omissions or capture
diagnostics for the invocation tested. The projected graphs contain 13 nodes / 8
edges before and 17 nodes / 11 edges after; these are graph projection counts,
not counts of user-defined layers.

Capture records with local paths remain outside this public repository. No
packages, saved profiles, telemetry settings or user Python environments were
changed. No browser, native application or installed-extension UI was opened.
Rendered graph appearance, editor interaction and opening through VS Code remain
unverified by this headless check.
