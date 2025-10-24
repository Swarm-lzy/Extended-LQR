#pragma once
#include <vector>
#include <Eigen/Dense>
#include <stdexcept>
#include <cassert>
#include "isolver_backend.hpp"

// 使用Schur Complement的方法解决Extended LQR对应的KKT方程组

namespace ocp {

struct StageU {
  Eigen::MatrixXd U11, U12, U22;
};


class SchurBackend final : public ISolverBackend {

public:
    const char* name() const override
    {
        return "SchurBackend"
    }

    void set_options(const Options& opt) override{
        opt_ = opt;
    }

    void clear_cache() override
    {

    }

    Status solve_unconstrained(const Problem& P, Solution& sol) override;

private:
    Options opt_;

    // 内部工具
    static void validate(const Problem& P);
    static double objective_value(const Problem& P, const Primal& prim);

    // 右乘上三角逆：返回 X = M * U^{-1}，U 为上三角
    static Eigen::MatrixXd right_mult_Uinv(const Eigen::MatrixXd& M, const Eigen::MatrixXd& U);

    struct StageU {
        Eigen::MatrixXd U11, U12, U22; // H = [Q S^T; S R] = U^T U 的上三角分块
    };
}




}