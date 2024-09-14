# Simple make file to store commands that are used often:

########################################################################################
# installation targets:

# The 'install' target runs the Python setup script to install the package
# for the current user. It uses the '--user' flag to install the package
# in the user-specific site-packages directory, avoiding the need for
# administrative privileges.
install:
	@python setup.py install --user

########################################################################################
# test targets:

# This Makefile target runs the unit tests for the tensiometer project.
# It uses Python's unittest module to discover and execute all test cases
# located in the 'tensiometer/tests' directory.
test:
	@python -m unittest discover tensiometer/tests

# Runs a specific test file using Python's unittest module.
# Usage: make test_file file=<test_filename>
test_file:
	@python -m unittest tensiometer/tests/${file}

# Generates a coverage report for the project.
# This target runs the `coverage report` command, which displays a summary
# of the code coverage for the tests in the project.
coverage_report:
	@coverage report

# This target runs unit tests with coverage analysis.
# It uses the 'coverage' tool to execute tests found in the 'tensiometer/tests' directory.
# The '-vvv' flag provides verbose output, and the '-f' flag stops on the first failure.
# After running the tests, it generates a coverage report.
test_with_coverage:
	@coverage run -m unittest discover tensiometer/tests -vvv -f
	@coverage report

########################################################################################
# example targets:

# This target navigates to the 'docs/example_notebooks' directory and executes all Jupyter 
# notebooks (*.ipynb) found in that directory. 
# The executed notebooks are then converted to HTML format. 
# The execution timeout is set to unlimited.
run_examples:
	@cd docs/example_notebooks && \
	for i in *.ipynb ; do \
		jupyter nbconvert --execute --to html $$i --ExecutePreprocessor.timeout=-1; \
	done;

# This target converts all Jupyter notebooks (*.ipynb) in the docs/example_notebooks
# directory to HTML format using nbconvert. It changes the directory to docs/example_notebooks,
# iterates over each notebook file, and converts it to HTML.
prepare_examples:
	@cd docs/example_notebooks && \
	for i in *.ipynb ; do \
		jupyter nbconvert --to html $$i; \
	done;

# The 'clean examples' target navigates to the 'docs/example_notebooks' directory
# and clears the output of all Jupyter notebooks (*.ipynb) in place using 'jupyter nbconvert'.
clean_examples:
	@cd docs/example_notebooks && \
	for i in *.ipynb ; do \
		jupyter nbconvert --clear-output --inplace $$i; \
	done;

########################################################################################
# documentation targets:

# Generates HTML documentation using Sphinx.
# Runs `sphinx-build` with the source directory `docs/source` and outputs to the `build` directory.
documentation:
	@sphinx-build -b html docs/source build

########################################################################################
# release targets:

# This Makefile target 'release' is used to package and upload the project to PyPI.
# It performs the following steps:
# 1. Creates source and binary distributions using 'setup.py'.
# 2. Uploads the generated distributions to the Python Package Index (PyPI) using 'twine'.
release:
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload --repository pypi dist/*

########################################################################################
# clean targets:

# The 'clean' target removes build artifacts and temporary files.
# It deletes the 'tensiometer.egg-info' directory, 'build' and 'dist' directories,
# the '.coverage' file, and any '__pycache__' directories within the 'tensiometer' 
# and 'docs/example_notebooks' directories.
clean:
	@rm -rf tensiometer.egg-info
	@rm -rf build dist
	@rm -f .coverage
	@rm -rf tensiometer/__pycache__ \
			tensiometer/*/__pycache__ \
			docs/example_notebooks/__pycache__
