#include <pybind11/pybind11.h>
#include <iostream>

namespace accelerated_functions {
    int add(int i, int j) {
        std::cout << "Executing accelerated_functions::add()..." << std::endl;
        return i + j;
    }
}

PYBIND11_MODULE(accelerated_functions_module, module) {
    module.def("add", &accelerated_functions::add, "A function that adds two numbers");
}