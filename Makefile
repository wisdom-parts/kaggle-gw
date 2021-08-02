MODULE := preprocess

run:
	@python -m $(MODULE)

test:
	@python -m pytest

clean:
	rm -rf .pytest_cache .coverage .pytest_cache coverage.xml

.PHONY: clean test
