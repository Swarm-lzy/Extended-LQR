#pragma once
#include <Eigen/Dense>  // 如果不用Eigen也可以，保证传入的data()是列主序即可
#include <vector>
#include <stdexcept>
#include <string>
#include <cstddef>

/*
列主序 → 右乘更快”，“行主序 → 左乘更快 Eigen 和 LAPACK都是列主序
列主序 (Col-major)：

在内存里，列是连续的。

如果你写成 右乘 B（即 C = A * B，每次从 B 取一列），那么 BLAS 读取 B 的一列时就是 连续内存块，效率最高。

所以 → 列主序下，右乘矩阵更 cache-friendly。

行主序 (Row-major)：

在内存里，行是连续的。

如果你写成 左乘 A（即 C = A * B，每次从 A 取一行），那么 BLAS 读取 A 的一行时就是 连续内存块。

所以 → 行主序下，左乘矩阵更快。
*/
// BLAS / LAPACK 头文件
extern "C" {
    #include <cblas.h>
    #include <lapacke.h>
}

namespace mpc
{
struct RiccatiProblem 
{
    // 维度
    int nx = 0; // 状态维度
    int nu = 0; // 控制维度
    int N = 0;  // 时域

    // 必需 A_n, B_n, Q_n, R_n, 以及终端 (P_N, p_N)
    std::vector<Eigen::MatrixXd> A; 
    std::vector<Eigen::MatrixXd> B;
    std::vector<Eigen::MatrixXd> Q;
    std::vector<Eigen::MatrixXd> R;

    // 可选：S_n, q_n, s_n, b_n（若缺省则视为 0）
    std::vector<Eigen::MatrixXd> S;  // [N]  each nu x nx   (注意论文里 S_n 在代价里与 x,u 交叉项对应)
    std::vector<Eigen::VectorXd> q;  // [N]  each nx
    std::vector<Eigen::VectorXd> s;  // [N]  each nu
    std::vector<Eigen::VectorXd> b;  // [N]  each nx

    // 终端项
    Eigen::MatrixXd P_N;  // nx x nx (对称半正定/正定)
    Eigen::VectorXd p_N;  // nx

    // 初始状态（用于前向模拟；若不需要前向模拟，可不设置）
    Eigen::VectorXd x0;   // nx
};


struct RiccatiResult 
{
    // 反馈/前馈
    std::vector<Eigen::MatrixXd> K; 
    std::vector<Eigen::VectorXd> k;

    // 值函数系数
    std::vector<Eigen::MatrixXd> P;  // [N+1], P[n] 为时刻 n 的 P_n，P[N]=P_N
    std::vector<Eigen::VectorXd> p;  // [N+1], p[n] 为时刻 n 的 p_n，p[N]=p_N

    // 前向状态与控制
    std::vector<Eigen::VectorXd> x;
    std::vector<Eigen::VectorXd> u;
};

struct RiccatiOptions
{
    bool do_forward_pass =  true;   // 是否进行前向模拟,生成{x, u}
    bool check_spd = true;  // dpotrf失败时报错
    bool use_symmetry = true;   // 利用P的对称性 algorithm 4
    bool force_chol_P = false;  // 强制对P_{n+1}做chol 并按上三角因子计算
};


// 基于LAPACKE/BLAS的Riccati递推求解器
class RiccatiSolver
{
public:
    RiccatiSolver() = default;

    RiccatiResult solve(const RiccatiProblem& prob, const RiccatiOptions& opt={});

private:
    static inline void assert_size(bool cond, const std::string& msg)
    {
        if (!cond) throw std::invalid_argument(msg);
    }

    // 小工具：Y = alpha*A*B + beta*Y （列主序）
  static void gemmNN(int m, int n, int k,
                     const double* A, int lda,
                     const double* B, int ldb,
                     double* Y, int ldy,
                     double alpha = 1.0, double beta = 0.0);

  static void gemmTN(int m, int n, int k,
                     const double* A, int lda, // A^T 参与
                     const double* B, int ldb,
                     double* Y, int ldy,
                     double alpha = 1.0, double beta = 0.0);

  static void gemmNT(int m, int n, int k,
                     const double* A, int lda,
                     const double* B, int ldb, // B^T 参与
                     double* Y, int ldy,
                     double alpha = 1.0, double beta = 0.0);

  // y = alpha*A*x + beta*y
  static void gemvN(int m, int n,
                    const double* A, int lda,
                    const double* x, int incx,
                    double* y, int incy,
                    double alpha = 1.0, double beta = 0.0);

  static void gemvT(int m, int n,
                    const double* A, int lda,
                    const double* x, int incx,
                    double* y, int incy,
                    double alpha = 1.0, double beta = 0.0);

  // 三角求解：解 U * X = B 或 U^T * X = B（矩阵右侧）
  static void trsm_left_upper(char trans, int m, int n,
                              const double* U, int ldu,
                              double* B, int ldb);

  // 三角求解向量：解 U * x = b 或 U^T * x = b
  static void trsv_upper(char trans, int n,
                         const double* U, int ldu,
                         double* x, int incx);


    /********************************** */
    // 新增封装，syrk / trmm / trmv
    // 对称 rank-k: C := alpha * op(A) * op(A)' + beta * C (只更新上三角)
    static void syrk_upper(char trans, int n, int k,
                           const double* A, int lda,
                           double* C, int ldc,
                            double alpha = 1.0, double beta = 0.0);
    
    // 三角乘: B := op(U) * B (U 上三角)
    static void trmm_left_upper(char transA, bool transpose_on_tri,
                                int m, int n,
                                const double* U, int ldu,
                                double* B, int ldb);

    // 对称乘向量: y := alpha * Sym(U) * x + beta * y (仅用上三角)
    static void trmv_upper(char trans, int n,
                            const double* U, int ldu,
                            double* x, int incx);

    // 新增封装 
    static void symv_upper(int n,
                            const double* UPU, int ldu,  // 对称矩阵，仅用上三角 
                            const double* x, int incx,
                            double* y, int incy,
                            double alpha = 1.0, double beta = 0.0);

    static void symm_left_upper(int n, int k,
                                const double* P, int ldP,  // P: n n 对称，仅用上三角
                                const double* M, int ldM,   // M: n k
                                double* Y, int ldy,         // Y: P * M
                                double alpha = 1.0, double beta = 0.0
                                );
                            
};



}
 