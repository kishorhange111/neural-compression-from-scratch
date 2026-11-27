"""The corpus: the Python standard library's own source, split into train and test by file.

Every Python install ships it, so nothing is downloaded. Files go to the test split by
a hash of their relative path, so no file is in both splits. The numbers in the README
come from CPython 3.13's ``Lib``. Another version gives a slightly different corpus.
"""

from __future__ import annotations

import os
import sysconfig
import zlib


def stdlib_files(root=None):
    root = root or sysconfig.get_paths()["stdlib"]
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in ("site-packages", "__pycache__"))
        for f in sorted(files):
            if f.endswith(".py"):
                p = os.path.join(d, f)
                out.append((os.path.relpath(p, root).replace(os.sep, "/"), p))
    return out


def split(root=None, test_every=20):
    """(train bytes, test bytes, test file names): one file in ``test_every`` is held out."""
    train, test, names = [], [], []
    for rel, path in stdlib_files(root):
        with open(path, "rb") as f:
            data = f.read()
        if zlib.crc32(rel.encode()) % test_every == 0:
            test.append(data)
            names.append(rel)
        else:
            train.append(data)
    return b"".join(train), b"".join(test), names
