SHELL=/bin/sh

install:
	pip install -r requirements.txt

docs:
	cd ./docs && $(MAKE)

test:
	python3 tests.py

.PHONY: install docs test
