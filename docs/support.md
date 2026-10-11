# Get help with tensorViz

First check the [setup and troubleshooting guide](https://tensorviz.dev/install.html#support).
For Remote SSH, confirm that the extension and the selected model Python are on
the SSH host, following the [Remote SSH guide](https://tensorviz.dev/install.html#remote).

## Public issues

This repository accepts questions and reproducible reports about the public
examples or using tensorViz. It does not provide access to the private extension
source. A report is not a promise of a response time or a supported platform.

Useful details are:

- tensorViz and VS Code versions;
- operating system and architecture;
- whether the workspace is local or Remote SSH;
- model Python and PyTorch versions;
- the action you took, expected result and actual result;
- a minimal example you have permission to publish, or whether an example in this
  repository reproduces the problem.

The issue forms ask for these details. If you only have private source, start
with a description and a small synthetic example instead of uploading it.

## What not to publish

Do not include tokens, passwords, environment files, connection URLs, raw logs,
analytics installation IDs, private repository URLs, personal information,
proprietary model code, checkpoints, tensors or datasets.

Graph exports and screenshots can contain file paths, source details, model
structure, comments and tensor information. Review them before sharing. Remove
private information, and do not upload an entire workspace or unreviewed archive.

If an error message includes a sensitive path or value, replace that portion with
a neutral label such as `<project>` and say that it was redacted. Keep the relevant
error type and steps to reproduce.

For privacy, analytics-deletion or security questions that should not be public,
contact [tensorviz.app@gmail.com](mailto:tensorviz.app@gmail.com). Do not send
passwords or API keys. Explain the issue first so that any follow-up can be scoped.

## Product behavior to keep in mind

- A successful capture covers one invocation. Test other inputs and branches as
  needed for your model.
- Source comparison does not run both revisions or prove model correctness.
- A failed capture may arise from constructor setup, inputs, model dependencies or
  unsupported execution behavior; report the smallest reproducible case.
- Optional usage analytics are not required to use tensorViz or receive help.

The **Optional analytics** section of the
[Marketplace listing](https://marketplace.visualstudio.com/items?itemName=tensorviz.tensorviz-vscode)
and the installed extension's `telemetry.json` describe the current collection
controls. Never put an analytics identity in a public issue.
