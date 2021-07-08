#!/bin/bash

curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

source ~/.zshrc
# source ~/.bashrcp

# pip3 install -r requirements.txt

# sudo apt-get install unzip
unzip -o data/dataset.zip -d data/
ls data/

poetry install
poetry run python3 src/prepare_data.py data/mnist_train.csv data/mnist_test.csv
ls data/prepared/

poetry run python3 src/train_mlp.py data/prepared mnist_mlp.joblib accuracy.json loss_history.json