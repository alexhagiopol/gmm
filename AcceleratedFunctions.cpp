#include <pybind11/pybind11.h>
#include <iostream>

namespace accelerated_functions {
    int add(int i, int j) {
        std::cout << "Executing accelerated_functions::add()..." << std::endl;
        return i + j;
    }
}