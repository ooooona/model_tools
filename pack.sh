#!/bin/bash

if [ ! -f ~/.pypirc ]; then
  echo "no ~/.pypirc, pls add manually"
  exit 1
fi

[ -d "dist" ] && rm -rf "dist"
[ ! -f "README.md" ] && echo "torch2onnx" > "README.md"

# check
python setup.py check
if [ $? -ne 0 ]; then
  echo "setup.py check failed!"
  exit 1
fi
# pack
python setup.py sdist
# upload
twine upload --verbose dist/*
