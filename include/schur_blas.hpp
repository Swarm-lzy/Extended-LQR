#pragma once
#include <Eigen/Dense>
#include <vector>

namespace ocp::schur_blas {

struct Inputs {
    int N, nx, nu;
    std::vector<Eigen::MatrixXd> Q, S, R, A, B;
    std::vector<Eigen::VectorXd> q, s, b;
    Eigen::MatrixXd QN;
    Eigen::VectorXd qN, x0;

};

struct Outputs {
    std::vector<Eigen::VectorXd> x; // 0,... N
    std::vector<Eigen::VectorXd> u; // 0,...N-1
};






}