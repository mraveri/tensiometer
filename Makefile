# Simple make file to store commands that are used often:

install:
	@python setup.py install --user

test:
	@python -m unittest discover tensiometer/tests

coverage_report:
	@coverage report

test_with_coverage:
	@coverage run -m unittest discover tensiometer/tests -vvv -f
	@coverage report

run_examples:
	@cd docs/example_notebooks && \
	for i in *.ipynb ; do \
		jupyter nbconvert --execute --to html $$i --ExecutePreprocessor.timeout=-1; \
	done;

prepare_examples:
	@cd docs/example_notebooks && \
	for i in *.ipynb ; do \
		jupyter nbconvert --to html $$i; \
	done;

documentation:
	@sphinx-build -b html docs/source build

release:
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload --repository pypi dist/*

clean:
	@rm -rf build dist
	@rm -f .coverage
