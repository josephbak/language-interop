CXX = clang++
CXXFLAGS = -std=c++17 -O2

PYTHON ?= python3
PYTHON_CONFIG ?= $(PYTHON)-config

PYTHON_CFLAGS = $(shell $(PYTHON_CONFIG) --cflags)
SUFFIX = $(shell $(PYTHON_CONFIG) --extension-suffix)

TARGET = autodiff$(SUFFIX)
SRC = autodiff.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -shared -fPIC -undefined dynamic_lookup \
		$(PYTHON_CFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	PYTHONPATH=. $(PYTHON) test.py

clean:
	rm -f $(TARGET)

.PHONY: all run clean