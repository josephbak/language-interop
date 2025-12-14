CXX = clang++

# Use PYTHON env var, or fall back to python3
PYTHON ?= python3
PYTHON_CONFIG ?= $(PYTHON)-config

PYTHON_CFLAGS = $(shell $(PYTHON_CONFIG) --cflags)
SUFFIX = $(shell $(PYTHON_CONFIG) --extension-suffix)

TARGET = hello$(SUFFIX)
SRC = hello.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) -shared -fPIC -undefined dynamic_lookup \
		$(PYTHON_CFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	PYTHONPATH=. $(PYTHON) test.py

clean:
	rm -f $(TARGET)

.PHONY: all run clean