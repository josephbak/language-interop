#include <Python.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

// Helper to read Python int
long py_get_long(PyObject* globals, const char* name) {
    PyObject* obj = PyDict_GetItemString(globals, name);
    if (!obj) {
        std::cerr << "Missing: " << name << std::endl;
        return 0;
    }
    return PyLong_AsLong(obj);
}

// Helper to read Python float
double py_get_double(PyObject* globals, const char* name) {
    PyObject* obj = PyDict_GetItemString(globals, name);
    if (!obj) {
        std::cerr << "Missing: " << name << std::endl;
        return 0.0;
    }
    return PyFloat_AsDouble(obj);
}

// Print a small region of the grid
void print_grid(const std::vector<std::vector<double>>& grid, int step) {
    std::cout << "\n=== Step " << step << " ===" << std::endl;
    
    int h = grid.size();
    int w = grid[0].size();
    
    // Print center 10x10 region
    int start_y = std::max(0, h/2 - 5);
    int start_x = std::max(0, w/2 - 5);
    
    for (int y = start_y; y < start_y + 10 && y < h; y++) {
        for (int x = start_x; x < start_x + 10 && x < w; x++) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(1) << grid[y][x];
        }
        std::cout << std::endl;
    }
}

int main() {
    Py_Initialize();

    // Load config
    FILE* fp = fopen("config.py", "r");
    if (!fp) {
        std::cerr << "Cannot open config.py" << std::endl;
        return 1;
    }
    PyRun_SimpleFile(fp, "config.py");
    fclose(fp);

    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* globals = PyModule_GetDict(main_module);

    // Read parameters
    int width = py_get_long(globals, "grid_width");
    int height = py_get_long(globals, "grid_height");
    double alpha = py_get_double(globals, "diffusion_rate");
    int steps = py_get_long(globals, "num_steps");
    int src_x = py_get_long(globals, "heat_source_x");
    int src_y = py_get_long(globals, "heat_source_y");
    double src_temp = py_get_double(globals, "heat_source_temp");
    int print_every = py_get_long(globals, "print_every");

    std::cout << "Heat Diffusion Simulation" << std::endl;
    std::cout << "Grid: " << width << "x" << height << std::endl;
    std::cout << "Diffusion rate: " << alpha << std::endl;
    std::cout << "Steps: " << steps << std::endl;

    // Initialize grid
    std::vector<std::vector<double>> grid(height, std::vector<double>(width, 0.0));
    std::vector<std::vector<double>> next_grid(height, std::vector<double>(width, 0.0));

    // Set initial heat source
    grid[src_y][src_x] = src_temp;

    // Run simulation (2D heat equation with finite differences)
    for (int step = 0; step <= steps; step++) {
        if (step % print_every == 0) {
            print_grid(grid, step);
        }

        // Compute next state
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                double laplacian = grid[y+1][x] + grid[y-1][x] + 
                                   grid[y][x+1] + grid[y][x-1] - 
                                   4.0 * grid[y][x];
                next_grid[y][x] = grid[y][x] + alpha * laplacian;
            }
        }

        std::swap(grid, next_grid);
        
        // Keep heat source constant
        grid[src_y][src_x] = src_temp;
    }

    Py_Finalize();
    return 0;
}