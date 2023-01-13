#include <pybind11/pybind11.h>
#include <iostream>

int add(int i, int j) {
    std::cout << "Executing accelerated_functions::add()..." << std::endl;
    return i + j;
}

PYBIND11_MODULE(accelerated_functions, module) {
    module.def("add", &add, "A function that adds two numbers");
}