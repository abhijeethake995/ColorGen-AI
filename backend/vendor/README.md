# Vendored DDColor inference architecture

This directory contains only the upstream files needed to instantiate DDColor-T and its dependencies. Source: [piddnad/DDColor](https://github.com/piddnad/DDColor), downloaded 2026-10-24. The upstream Apache-2.0 license is included as `DDCOLOR-LICENSE.txt`.

`ddcolor/model.py` and `basicsr/archs/ddcolor_arch_utils/` retain the upstream architecture. The model file's four `basicsr` imports were changed to package-relative imports so this subset runs inside ColorGen without installing the full BasicSR project or compiling its optional extensions.

The matching official pretrained checkpoint is stored outside version control under `models/ddcolor_paper_tiny.bin`. The app loads it with `weights_only=True` and strict state-dict matching.
