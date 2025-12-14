#include <Python.h>
#include <iostream>

int main() {
    // Initialize the Python interpreter
    Py_Initialize();

    if (!Py_IsInitialized()) {
        std::cerr << "Failed to initialize Python" << std::endl;
        return 1;
    }

    std::cout << "Python interpreter initialized" << std::endl;
    std::cout << "Python version: " << Py_GetVersion() << std::endl;

    // Run some Python code
    PyRun_SimpleString("print('Hello from Python inside C++!')");
    PyRun_SimpleString("x = 40 + 2");
    PyRun_SimpleString("print(f'The answer is {x}')");

    // Shutdown
    Py_Finalize();
    std::cout << "Python interpreter finalized" << std::endl;

    return 0;
}