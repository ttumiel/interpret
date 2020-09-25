all:
	make install

install:
	pip install -e .

clean:
	rm -rf build dist

test:
	pytest

build:
	python setup.py sdist bdist_wheel
	tar tzf dist/interpret-pytorch*.tar.gz
	twine check dist/*

publish:
	twine upload dist/*

.PHONY: all install clean test build publish
