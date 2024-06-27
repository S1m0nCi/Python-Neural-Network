PYTHONPATH=.  # Add current directory to search path for modules (optional)

# Define your testing command
TEST_COMMAND = python -m unittest discover -v


# Run tests by default
all: test

test:
  @echo "Running tests..."
  $(TEST_COMMAND)

clean:
  @echo "Cleaning up..."
  rm -rf __pycache__* tests/*.pyc

install:
	@echo "Installing project dependencies..."
	poetry install
	poetry lock
	
.PHONY: all test clean install
