# Common developer tasks: install, test, docs, release, and cleanup helpers.
# Use `make test_file file=<name>` to run a single test module.
.PHONY: install test tests test_file coverage_report test_with_coverage run_examples coalesce_examples prepare_examples clean_examples documentation release clean

################################################################################
# Installation
install:
	@python3 -m pip install --user .

# Install only dependencies from requirements.txt.
install_requirements:
	@python3 -m pip install --user -r requirements.txt

install_dev:
	@python3 -m pip install --user -e .[dev]

################################################################################
# Tests
test:
	@python3 -m pytest -q

tests: test

# Run a single test module by passing file=<module_name>.
test_file:
	@python3 -m pytest -q tensiometer/tests/${file}

# Show coverage summary from the last run.
coverage_report:
	@python3 -m coverage report

# Run tests with coverage collection.
test_with_coverage:
	@python3 -m pytest --cov=tensiometer --cov-report=term-missing --cov-config=.coveragerc

################################################################################
# Examples
# Execute all notebooks in place, then tidy their outputs and export to HTML.
run_examples:
	@cd docs/example_notebooks && \
	for i in *.ipynb ; do \
		jupyter nbconvert --to notebook --execute --inplace $$i || exit 1; \
	done;
	@$(MAKE) prepare_examples

# Merge fragmented stream outputs (e.g. tqdm progress bars) left by nbconvert --execute.
# This needs its own pass: nbconvert coalesces streams before executing, not after.
coalesce_examples:
	@cd docs/example_notebooks && \
	for i in *.ipynb ; do \
		jupyter nbconvert --coalesce-streams --inplace $$i; \
	done;

prepare_examples: coalesce_examples
	@cd docs/example_notebooks && \
	for i in *.ipynb ; do \
		jupyter nbconvert --to html $$i; \
	done;

clean_examples:
	@cd docs/example_notebooks && \
	for i in *.ipynb ; do \
		jupyter nbconvert --clear-output --inplace $$i; \
	done;

################################################################################
# Docs
documentation:
	@sphinx-build -b html docs/source build

################################################################################
# Release
release:
	@python3 -m build .
	@python3 -m twine upload --repository pypi dist/* --verbose

################################################################################
# Cleanup
clean:
	@rm -rf tensiometer.egg-info
	@rm -rf build dist
	@rm -f .coverage
	@rm -rf .pytest_cache
	@rm -rf tensiometer/__pycache__ \
			tensiometer/*/__pycache__ \
			docs/example_notebooks/__pycache__
