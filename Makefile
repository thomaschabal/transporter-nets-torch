all: install
.PHONY: all

install:
	conda create --name ravens_torch python=3.7 -y; \
	conda activate ravens_torch; \
	pip install -r requirements.txt; \
	python setup.py install --user

demos:
	python ravens_torch/demos.py

train:
	python ravens_torch/train.py

test:
	python ravens_torch/test.py

plot:
	python ravens_torch/plot.py
