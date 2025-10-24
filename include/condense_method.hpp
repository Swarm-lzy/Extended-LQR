#pragma once
#include <Eigen/Dense>
#include <vector>
#include <optional>
#include <string>
#include <stdexcept>

// Condensing method for Extended LQ Control  (keeps the same public interface style
// as the Schur-complement solver: Problem / Options / Factorization / Result + Solver)

// Cost: sum_{n=0}^{N-1} [ x_n^T Q_n x_n + 2 x_n^T S_n u_n + u_n^T R_n u_n
// + 2 q_n^T x_n + 2 s_n^T u_n ] + x_N^T Q_N x_N + 2 q_N^T x_N
// Dynamics: x_{n+1} = A_n x_n + B_n u_n + b_n, with x_0 given.

extern "C"
{
    #include <cblas.h>
    #include <lapacke.h>
}


namespace mpc {

struct CondenseOptions
{
    bool symmetrize_hessian = true; 
    bool check_spd = true;
    bool do_forward_pass = true;
    double diag_regularization = 1e-9;  
};

struct CondenseProblem
{
    // 维度
    int nx = 0; // 状态维度
    int nu = 0; // 控制维度
    int N = 0;  // 时域
    
    // 动力学/代价
    std::vector<Eigen::MatrixXd> A; // [N] nx nx
    std::vector<Eigen::MatrixXd> B; // [N] nx nu
    std::vector<Eigen::VectorXd> b; // [N] nx 

    std::vector<Eigen::MatrixXd> Q; // [N] nx nx
    std::vector<Eigen::MatrixXd> R; // [N] nu nu
    std::vector<Eigen::MatrixXd> S; // [N] nu nx
    std::vector<Eigen::VectorXd> q; // [N] nx
    std::vector<Eigen::VectorXd> s; // [N] nu

    Eigen::MatrixXd QN; // nx nx
    Eigen::VectorXd qN; // nx

    // 初始状态 (用于 forward-pass)
    Eigen::VectorXd x0; // nx
};

struct CondenseFactorization
{
    // 凝聚后的密集 QP: 1/2 U^T Hc U + gc^T U + const
    Eigen::MatrixXd Hc;     // (N *nu N * nu)
    Eigen::VectorXd gc;     // N * nu
    std::optional<Eigen::LLT<Eigen::MatrixXd>> llt; // Cholesky因子

    // 中间量
    Eigen::MatrixXd Gu; // 
    Eigen::VectorXd Gbb;
    Eigen::MatrixXd Qbar;
    Eigen::MatrixXd Sbar;
    Eigen::MatrixXd Rbar;
    Eigen::VectorXd qbar;
    Eigen::VectorXd sbar;
};


struct CondenseResult {
    std::vector<Eigen::VectorXd> x; // x_0 ... x_N
    std::vector<Eigen::VectorXd> u; // u_0 .... u_{N-1}
    double cost = 0.0;  

};

class CondenseSolver{

public:
        CondenseSolver() = default;

        CondenseResult solve(const CondenseProblem& prob, const CondenseOptions& opt = {});

        // 显式两阶段接口
        void factor(const CondenseProblem& prob, CondenseFactorization& F, const CondenseOptions& opt = {});

        void solve_with_factor(const CondenseProblem& prob, const CondenseOptions& opt,
                            CondenseFactorization& F, CondenseResult& res);

private:
    static inline void assert_size(bool cond, const std::string& msg)
    {
        if (!cond)
        {
            throw std::invalid_argument(msg);
        }
    }
    
    // ====== BLAS/LAPACK wrapper 接口 ========== 
    static void gemmNN(int m, int n, int k,
                        const double* A, int lda,
                        const double* B, int ldb,
                        double* C, int idc,
                        double alpha = 1.0, double beta = 0.0);
    static void gemmTN(int m, int n, int k,
                        const double*AT, int lda,
                        const double* B, int ldb,
                        double* C, int ldc,
                        double alpha = 1.0, double beta = 0.0);
    static void gemmNT(int m, int n, int k,
                        const double* A, int lda,
                        const double* BT, int ldb,
                        double* C, int ldc,
                        double alpha = 1.0, double beta = 0.0);
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
    static void syrk_upper(char trans, int n, int k,
                            const double* A, int lda,
                            double* C, int ldc,
                            double alpha = 1.0, double beta = 0.0);
    static void trsm_left_upper(char trans, int m, int n, 
                                const double* U, int ldu,
                                double* B, int ldb);
    static void trsv_upper(char trans, int n,
                            const double* U, int ldu,
                            double* x, int incx);
    static void symv_upper(int n, const double* S,
                            int ldS, const double* x,
                            int incx, double* y, int incy, double alpha = 1.0, double beta = 0.0);

    // builders
    static void build_Gamma(const CondenseProblem& P, Eigen::MatrixXd& Gu, Eigen::VectorXd& Gbb);
    static void blkdiag_Qbar(const CondenseProblem& P, Eigen::MatrixXd& Qbar);
    static void blkdiag_Sbar(const CondenseProblem& P, Eigen::MatrixXd& Sbar);
    static void blkdiag_Rbar(const CondenseProblem& P, Eigen::MatrixXd& Rbar);
    static void stack_qbar(const CondenseProblem& P, Eigen::VectorXd& qbar);
    static void stack_sbar(const CondenseProblem& P, Eigen::VectorXd& sbar);

    static void check_dims(const CondenseProblem& P);

};

struct Problem {
    int N = 0;  // horizon
    int nx = 0; // state dim
    int nu = 0; // control dim

    Eigen::VectorXd x0; //nx
    std::vector<Eigen::MatrixXd> Q; // N blocks (nx nx)
    std::vector<Eigen::MatrixXd> S; // N blocks (nu nx)
    std::vector<Eigen::MatrixXd> R; // N blocks (nu nu)
    std::vector<Eigen::VectorXd> q; // N blocks (nx)
    std::vector<Eigen::VectorXd> s; // N blocks (nu) 
    std::vector<Eigen::MatrixXd> A; // N blocks (nx nx)
    std::vector<Eigen::MatrixXd> B; // N blocks (nx nu)
    std::vector<Eigen::VectorXd> b; // N blocks (nx)

    Eigen::MatrixXd QN; // (nx nx)
    Eigen::VectorXd qN; // (nx)
};

struct Factorization {
    // condensed dense Hessian and its LL^T factor
    Eigen::MatrixXd Hc; // (N*nu N*nu)
    std::optional<Eigen::LLT<Eigen::MatrixXd>> llt; //Cholesky factor

    // reusable intermediates
    Eigen::MatrixXd Gu; // Γ_u (N*nx x N*nu)
    Eigen::VectorXd Gbb; // Γ_bb (N*nx)
    Eigen::MatrixXd Qbar; // blkdiag(Q_1..Q_N)
    Eigen::MatrixXd Sbar; // blkdiag(S_0..S_{N-1})
    Eigen::MatrixXd Rbar; // blkdiag(R_0..R_{N-1})
    Eigen::VectorXd qbar; // [q_1;..;q_N]
    Eigen::VectorXd sbar; // [s_0 + S_0 x0; s_1;..;s_{N-1}]
    Eigen::VectorXd gc; // linear term

};

struct Result {
    std::vector<Eigen::VectorXd> u; // u_0 .. u_{N-1}
    std::vector<Eigen::VectorXd> x; // x_0 .. x_N
    double cost = 0.0;
};







}