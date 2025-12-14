CXX = clang++

PYTHON ?= python3
PYTHON_CONFIG ?= $(PYTHON)-config

PYTHON_CFLAGS = $(shell $(PYTHON_CONFIG) --cflags)
SUFFIX = $(shell $(PYTHON_CONFIG) --extension-suffix)

TARGET = fastmath$(SUFFIX)
SRC = fastmath.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) -shared -fPIC -undefined dynamic_lookup \
		$(PYTHON_CFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	PYTHONPATH=. $(PYTHON) test.py

bench: $(TARGET)
	PYTHONPATH=. $(PYTHON) benchmark.py

clean:
	rm -f $(TARGET)

.PHONY: all run bench clean