#include <Eigen/Dense>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <string>

using EigenRowMajMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor>;

/// see compute_expsum_stable() in gmm_segmentation.py for documentation
void computeExpsumStable(
        const Eigen::Ref<EigenRowMajMatrixXd> intensities, 
        pybind11::list weights,
        pybind11::list means,
        pybind11::list stdevs,
        Eigen::Ref<EigenRowMajMatrixXd> expsum,
        Eigen::Ref<EigenRowMajMatrixXd> P,
        Eigen::Ref<EigenRowMajMatrixXd> P_max
        ) {
    // TODO
}

/// example function to test Python<->PyBind<->C++
void fill(Eigen::Ref<EigenRowMajMatrixXd> matrix, double value) {
    for (int r = 0; r < matrix.rows(); r++) {
        for (int c = 0; c < matrix.cols(); c++) {
            matrix(r, c) = value;
        }
    }
}

/// example function to test Python<->PyBind<->C++
int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(accelerated_functions, module) {
    module.def("add", &add, "A function that adds two numbers.");
    module.def("computeExpsumStable", &computeExpsumStable, "See compute_expsum_stable() in gmm_segmentation.py for documentation.");
    module.def("fill", &fill, "Set every value of a matrix to a given value.");
}
