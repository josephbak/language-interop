// 01-embedding/02-read-variables/main.cpp
#include <Python.h>
#include <iostream>
#include <string>

int main() {
    Py_Initialize();

    // Run the config file
    FILE* fp = fopen("config.py", "r");
    if (!fp) {
        std::cerr << "Cannot open config.py" << std::endl;
        return 1;
    }
    PyRun_SimpleFile(fp, "config.py");
    fclose(fp);

    // Get the __main__ module's namespace (where variables live)
    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* globals = PyModule_GetDict(main_module);

    // Read simulation_name (string)
    PyObject* name_obj = PyDict_GetItemString(globals, "simulation_name");
    std::string sim_name = PyUnicode_AsUTF8(name_obj);

    // Read num_iterations (int)
    PyObject* iter_obj = PyDict_GetItemString(globals, "num_iterations");
    long iterations = PyLong_AsLong(iter_obj);

    // Read time_step (float)
    PyObject* dt_obj = PyDict_GetItemString(globals, "time_step");
    double dt = PyFloat_AsDouble(dt_obj);

    // Read grid_size (tuple)
    PyObject* grid_obj = PyDict_GetItemString(globals, "grid_size");
    long grid_x = PyLong_AsLong(PyTuple_GetItem(grid_obj, 0));
    long grid_y = PyLong_AsLong(PyTuple_GetItem(grid_obj, 1));

    // Print from C++
    std::cout << "=== Configuration Loaded ===" << std::endl;
    std::cout << "Simulation: " << sim_name << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Time step:  " << dt << std::endl;
    std::cout << "Grid size:  " << grid_x << " x " << grid_y << std::endl;

    Py_Finalize();
    return 0;
}