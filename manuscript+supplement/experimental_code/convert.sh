#!/usr/bin/env bash

set -euxo pipefail

jupytext --to py *.ipynb
sed -i '' -e 's/%matplotlib inline//g' *.py
sed -i '' -e 's/plt.show()//g' *.py
mv *py scripts
