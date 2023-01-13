#include <pybind11/pybind11.h>
#include <iostream>
#include <string>

int add(int i, int j) {
    const int result = i + j;
    const std::string strFunctionSignature = "add(" + std::to_string(i) + "," + std::to_string(j) + ")";
    const std::string strResult = "=" + std::to_string(result);
    std::cout << "C++ code: executing " << strFunctionSignature << strResult << std::endl;
    return result;
}

PYBIND11_MODULE(accelerated_functions, module) {
    module.def("add", &add, "A function that adds two numbers");
}