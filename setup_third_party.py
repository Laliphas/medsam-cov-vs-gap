"""Fetch the official code at the commit we used and apply our patch.

    python setup_third_party.py

Result: third_party/2026-isbi-mayr (patched). On the first run of any script, the official code
asks for the cache / dataset / result / model folders and stores them in its config.json.
"""
import os
import subprocess

REPO = os.path.dirname(os.path.abspath(__file__))
URL = 'https://github.com/compai-lab/2026-isbi-mayr'
COMMIT = '6457686a2b90c921249d3fc2b9fc9096a03c6e75'
DEST = os.path.join(REPO, 'third_party', '2026-isbi-mayr')
PATCH = os.path.join(REPO, 'patches', 'official_changes.patch')


def git(*args, cwd=DEST):
    subprocess.run(['git', *args], cwd=cwd, check=True)


if __name__ == '__main__':
    if os.path.exists(DEST):
        raise SystemExit(f'{DEST} already exists')
    git('clone', URL, DEST, cwd=REPO)
    git('checkout', COMMIT)
    git('apply', PATCH)
    print(f'official code at {COMMIT[:7]} + patches/official_changes.patch -> {DEST}')
