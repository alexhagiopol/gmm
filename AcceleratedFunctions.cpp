#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <numbers>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using EigenRowMajMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor>;

/// see compute_expsum_stable() in gmm_segmentation.py for documentation
void computeExpsumStable(
        const Eigen::Ref<EigenRowMajMatrixXd> intensities, 
        std::vector<double> weightsList,
        std::vector<double> meansList,
        std::vector<double> stdevsList,
        Eigen::Ref<EigenRowMajMatrixXd> expsum,
        Eigen::Ref<EigenRowMajMatrixXd> P_2D,
        Eigen::Ref<EigenRowMajMatrixXd> P_max
        ) {
    // implement Equation 9 derived in Hagiopol paper
    const int K = weightsList.size();
    const int numRows = expsum.rows();
    const int numCols = expsum.cols();
    for (int k = 0; k < K; k++) {
        const double logWeightK = std::log(weightsList[k]);
        const double stdev = stdevsList[k];
        const double mean = meansList[k];
        // implement https://en.wikipedia.org/wiki/Gaussian_function
        const double multiplier = 1 / (stdev * std::sqrt(2 * std::numbers::pi));
        const double base = std::exp(-1 / (2 * stdev * stdev));
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                const double intensity = intensities(r, c);
                const double exponent = std::pow(intensity - mean, 2);
                const double gaussPdfValue = multiplier * std::pow(base, exponent);
                P_2D(r, c + k*numCols) = logWeightK + std::log(gaussPdfValue);
            }
        }
    }
    // implement Equation 10 derived in Hagiopol paper
    for (int r = 0; r < numRows; r++) {
        for (int c = 0; c < numCols; c++) {
            double maxValue = std::numeric_limits<double>::min();
            for (int k = 0; k < K; k++) {
                const double currentValue = P_2D(r, c + k*numCols);
                if (currentValue > maxValue) maxValue = currentValue;
            }
            P_max(r, c) = maxValue;
        }
    }
    // implement expsum calculation used in Equation 11 derived in Hagiopol paper
    for (int k = 0; k < K; k++) {
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                expsum(r, c) += std::exp(P_2D(r, c + numCols*k) - P_max(r, c)); 
            }
        }
    }
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
