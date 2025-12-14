CXX = clang++
CXXFLAGS = -std=c++17 -O2
PYTHON_CFLAGS = $(shell python3-config --cflags)
PYTHON_LDFLAGS = $(shell python3-config --ldflags --embed)

TARGET = simulation
SRC = simulation.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYTHON_CFLAGS) $(SRC) $(PYTHON_LDFLAGS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean