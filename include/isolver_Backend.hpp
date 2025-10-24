#pragma once
#include <string>
#include <memory>
#include <vector>
#include <optional>

#include "type.hpp"
#include "problem.hpp"

namespace ocp{

struct BackendStats {

};


struct ECQP {
// Quadratic term and linear term over stacked decision (z). For LQR this is
// typically block-diagonal H built from {Q,S,R} along the horizon.
Eigen::MatrixXd H; // SPD (or regularized SPD)
Eigen::VectorXd g; // linear term (can be zero in Newton steps)


// Equality constraints A z = b (dynamics + any additional equalities)
Eigen::MatrixXd A; // usually sparse/banded when built from OCP
Eigen::VectorXd b;


// Optional regularization (backends may add their own small regs)
double reg_primal{0.0};
double reg_dual{0.0};
};


class ISolverBackend {
public:
    virtual ~ISolverBackend() = default;

    virtual void set_options(const Options& opt) {
        (void)opt;
    }

    virtual Status solve_unconstrained(const Problem& p, Solution& sol) = 0;

    struct ECQPResult {
    Eigen::VectorXd dz; // primal step
    Eigen::VectorXd dl; // equality multiplier step
    Status status; // kOK if solved
    };

    // 从QP的角度解决LQR问题
    virtual ECQPResult solve_ecqp(const ECQP& sys) {
    return ECQPResult{Eigen::VectorXd(), Eigen::VectorXd(), Status{Status::kINTERNAL, "ECQP solve not implemented by backend"}};
    }

    // Optional hooks for factor caching across multiple calls (e.g., IPM predictor/ corrector)
virtual void clear_cache() {}


// Introspection
virtual const char* name() const = 0;
virtual BackendStats stats() const { return BackendStats{}; }


}



    


}