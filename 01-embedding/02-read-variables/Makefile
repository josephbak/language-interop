CXX = clang++
PYTHON_CFLAGS = $(shell python3-config --cflags)
PYTHON_LDFLAGS = $(shell python3-config --ldflags --embed)

TARGET = read_vars
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(PYTHON_CFLAGS) $(SRC) $(PYTHON_LDFLAGS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean