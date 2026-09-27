# Legacy v5 manifests

These are the v5 ("fat" bundle) manifests from before the models moved to v6 thin bundles.
They do NOT describe what is published on Hugging Face today, and `tt_kernel_manifest-1024.json`
still pins a withdrawn checkpoint, so don't package from them.

The published `episod/tt-tnt` and `episod/tt-tnt-1024` bundles are staged by
[`packaging/package-thin.sh`](../packaging/package-thin.sh), which carries every flag and the
extra files (mesh descriptor, chip pin) a v6 bundle needs. `tests/test_packaging_recipe.py`
checks that recipe; `tests/test_manifests.py` still checks these files' internal coherence.
