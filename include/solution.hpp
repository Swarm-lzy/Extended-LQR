#pragma once
// include/ocp/core/solution.hpp
#include <Eigen/Dense>
#include <vector>
#include <string>
#include "types.hpp"

namespace ocp {

struct Primal {
  // x_0..x_N, u_0..u_{N-1}
  std::vector<Eigen::VectorXd> x;  // size N+1
  std::vector<Eigen::VectorXd> u;  // size N
};

struct Dual {
  // 等式约束乘子（例如动力学的 π_1..π_N）
  std::vector<Eigen::VectorXd> lam_eq;   // size N
  // 不等式约束乘子（可选；若未用，保持为空）
  std::vector<Eigen::VectorXd> lam_ineq; // 按需要组织（逐阶段或整体）
};

struct Status {
  enum Code { kOK=0, kINVALID_INPUT, kSINGULAR, kMAX_ITERS, kNUMERICS, kINTERNAL } code{kOK};
  std::string msg;
  explicit operator bool() const { return code == kOK; }
};

struct Solution {
  Primal prim;
  Dual   dual;
  double cost{0.0};
  Status status;
  int    iters{0};
};

} // namespace ocp