# Checkpoints

Place trained checkpoints in this directory. Checkpoint files (`*.pth`) are ignored by Git by default so that large or unverified files are not committed accidentally.

Expected naming convention:

```text
checkpoints/
  uieb/physdsformer-tiny.pth
  uieb/physdsformer-tiny-last.pth
  lsui/physdsformer-tiny.pth
  euvp/physdsformer-tiny.pth
```

The regular file is the best validation checkpoint; `-last.pth` is the latest resumable training state. Each checkpoint should contain either a plain PyTorch state dictionary or a dictionary with a `state_dict` entry.
