#pragma once

namespace ocp{

struct LinearSolverOptions {
  enum class Backend { Riccati, Schur, Dense } backend{Backend::Riccati};
  bool   reuse_factors{true};   // 复用因子（适合多次求解/迭代）
  bool   equilibrate{false};    // 预处理/平衡（可选）
  double reg_primal{1e-9};      // 主变量正则（保证 SPD/数稳）
  double reg_dual{1e-12};       // 乘子正则（Schur/ECQP 中用）
};

struct QPSolverOptions {
  enum class Method { PDIPM, ActiveSet, ADMM } method{Method::PDIPM};
  int    max_iters{50};
  double rel_tol{1e-6};
  double abs_tol{1e-8};
  bool   warm_start{true};
};

struct Options {
  LinearSolverOptions lin;
  QPSolverOptions     qp;
  int verbose{0};
};
}