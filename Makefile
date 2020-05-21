# Simple make file to store commands that are used often:

test:
	@python -m unittest discover tensiometer/tests

test_with_coverage:
	@coverage run -m unittest discover tensiometer/tests
	@coverage report

coverage_report:
	@coverage report

doc:
	@sphinx-build -b html docs/source build

release:
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload --repository pypi dist/*

clean:
	@rm -rf build dist
	@rm -f .coverage
