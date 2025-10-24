#include <condense_method.hpp>
#include <algorithm>

/*
1. 构造状态展开矩阵Gamma_u, Gamma_bb  X = Gamma_u * U + Gamma_bb
2. 构造块对角矩阵Qbar，Sbar, Rbar, qbar, sbar   
3. 消元，把状态代入代价函数，得到只有U作为变量的函数 0.5* U^T * H * U + g^T U
4. 求无约束优化问题的解，计算Hessian H和梯度g
5. 对称正定性检验，Cholesky分解求最优控制序列 U = -H^{-1} * g
*/

namespace mpc
{
    using Eigen::MatrixXd; using Eigen::VectorXd;

    // ======= BLAS/LAPACK wrappers
    void CondenseSolver::gemmNN(int m,int n,int k,const double* A,int lda,const double* B,int ldb,double* C,int ldc,double alpha,double beta){
cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m,n,k, alpha, A,lda, B,ldb, beta, C,ldc);
}
void CondenseSolver::gemmTN(int m,int n,int k,const double* AT,int lda,const double* B,int ldb,double* C,int ldc,double alpha,double beta){
cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m,n,k, alpha, AT,lda, B,ldb, beta, C,ldc);
}
void CondenseSolver::gemmNT(int m,int n,int k,const double* A,int lda,const double* BT,int ldb,double* C,int ldc,double alpha,double beta){
cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m,n,k, alpha, A,lda, BT,ldb, beta, C,ldc);
}
void CondenseSolver::gemvN(int m,int n,const double* A,int lda,const double* x,int incx,double* y,int incy,double alpha,double beta){
cblas_dgemv(CblasColMajor, CblasNoTrans, m,n, alpha, A,lda, x,incx, beta, y,incy);
}
void CondenseSolver::gemvT(int m,int n,const double* A,int lda,const double* x,int incx,double* y,int incy,double alpha,double beta){
cblas_dgemv(CblasColMajor, CblasTrans, m,n, alpha, A,lda, x,incx, beta, y,incy);
}
void CondenseSolver::syrk_upper(char trans,int n,int k,const double* A,int lda,double* C,int ldc,double alpha,double beta){
CBLAS_TRANSPOSE T = (trans=='N'||trans=='n')? CblasNoTrans : CblasTrans;
cblas_dsyrk(CblasColMajor, CblasUpper, T, n,k, alpha, A,lda, beta, C,ldc);
}
void CondenseSolver::trsm_left_upper(char trans,int m,int n,const double* U,int ldu,double* B,int ldb){
CBLAS_TRANSPOSE T = (trans=='N'||trans=='n')? CblasNoTrans : CblasTrans;
cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, T, CblasNonUnit, m,n, 1.0, U,ldu, B,ldb);
}
void CondenseSolver::trsv_upper(char trans,int n,const double* U,int ldu,double* x,int incx){
CBLAS_TRANSPOSE T = (trans=='N'||trans=='n')? CblasNoTrans : CblasTrans;
cblas_dtrsv(CblasColMajor, CblasUpper, T, CblasNonUnit, n, U,ldu, x,incx);
}
void CondenseSolver::symv_upper(int n,const double* S,int ldS,const double* x,int incx,double* y,int incy,double alpha,double beta){
cblas_dsymv(CblasColMajor, CblasUpper, n, alpha, S,ldS, x,incx, beta, y,incy);
}

// builders 
void CondenseSolver::build_Gamma(const CondenseProblem& P, MatrixXd& Gu, VectorXd& Gbb)
{
    const int N = P.N, nx = P.nx, nu = P.nu;
    Gu.setZero(N*nx, N*nu);
    Gbb.setZero(N*nx);
    Gu.block(0,0,nx,nu) = P.B[0];
    Gbb.segment(0, nx) = P.A[0] * P.x0 + (P.b.empty() ? VectorXd::Zero(nx) : P.b[0]);
    for (int n = 1; n < N; ++n)
    {
        Gu.block(n*nx, 0, nx, n*nu) = P.A[n] * Gu.block((n-1)*nx, 0, nx, n*nu);
        Gu.block(n*nx, n*nu, nx, nu) = P.B[n];
        const VectorXd bn = P.b.empty() ? VectorXd::Zero(nx) : P.b[n];
        Gbb.segment(n*nx, nx) = P.A[n]*Gbb.segment((n-1)*nx, nx) + bn;
    }
}

void CondenseSolver::blkdiag_Qbar(const CondenseProblem& P, MatrixXd& Qbar)
{
    const int N = P.N, nx = P.nx;
    Qbar.setZero(N*nx, N*nx);
    for (int n=0; n<N; ++n)
    {
        Qbar.block(n*nx, n*nx, nx, nx) = P.Q[n+1];
    }
    Qbar.block((N-1)*nx, (N-1)*nx, nx, nx) = P.QN;
}

void CondenseSolver::blkdiag_Sbar(const CondenseProblem& P, MatrixXd& Sbar)
{
    const int N = P.N, nx = P.nx, nu = P.nu;
    Sbar.setZero(N*nu, N*nx);
    for (int n = 0; n<N; ++n)
    {
        if (!P.S.empty())   Sbar.block(n*nu, n*nx, nu, nx) = P.S[n];
    }
}
void CondenseSolver::blkdiag_Rbar(const CondenseProblem& P, MatrixXd& Rbar)
{
    const int N = P.N, nu = P.nu;
    Rbar.setZero(N*nu, N*nu);
    for(int n=0; n<N; ++n)
    {
        Rbar.block(n*nu, n*nu, nu, nu) = P.R[n];
    }
}

void CondenseSolver::stack_qbar(const CondenseProblem& P, VectorXd& qbar)
{
    const int N = P.N, nx = P.nx;
    qbar.setZero(N*nx);
    for(int n=0; n<N-1; ++n) qbar.segment(n*nx, nx) = P.q.empty() ? VectorXd::Zero(nx) : P.q[n+1];
    qbar.tail(nx) = P.qN;
}

void CondenseSolver::stack_sbar(const CondenseProblem& P, VectorXd& sbar)
{
    const int N=P.N, nx=P.nx, nu=P.nu; sbar.setZero(N*nu);
    // s0 + S0 x0
    if(P.s.empty()) sbar.segment(0,nu).setZero();
    else sbar.segment(0,nu) = P.s[0];
    if(!P.S.empty()) sbar.segment(0,nu).noalias() += P.S[0] * P.x0;
    for(int n=1;n<N;++n){
        if(P.s.empty()) sbar.segment(n*nu,nu).setZero();
        else sbar.segment(n*nu,nu) = P.s[n];
    }
}

void CondenseSolver::check_dims(const CondenseProblem& P)
{
    if(P.N<=0) throw std::invalid_argument("N must be > 0");
    auto szN=[&](size_t a){return (int)a==P.N;};
    if(!szN(P.A.size())||!szN(P.B.size())||!szN(P.Q.size())||!szN(P.R.size()))
        throw std::invalid_argument("A,B,Q,R must be size N");
    if((int)P.x0.size()!=P.nx) throw std::invalid_argument("x0 dim");
    if(P.QN.rows()!=P.nx||P.QN.cols()!=P.nx) throw std::invalid_argument("QN dim");
    if((int)P.qN.size()!=P.nx) throw std::invalid_argument("qN dim");
    for(int n=0;n<P.N;++n){
        if(P.A[n].rows()!=P.nx||P.A[n].cols()!=P.nx) throw std::invalid_argument("A dim");  
        if(P.B[n].rows()!=P.nx||P.B[n].cols()!=P.nu) throw std::invalid_argument("B dim");
        if(P.Q[n].rows()!=P.nx||P.Q[n].cols()!=P.nx) throw std::invalid_argument("Q dim");
        if(P.R[n].rows()!=P.nu||P.R[n].cols()!=P.nu) throw std::invalid_argument("R dim");
        if(!P.S.empty() && (P.S[n].rows()!=P.nu||P.S[n].cols()!=P.nx)) throw std::invalid_argument("S dim");
        if(!P.q.empty() && (int)P.q[n].size()!=P.nx) throw std::invalid_argument("q dim");
        if(!P.s.empty() && (int)P.s[n].size()!=P.nu) throw std::invalid_argument("s dim");
        if(!P.b.empty() && (int)P.b[n].size()!=P.nx) throw std::invalid_argument("b dim");
    }
}



// ========= factor / solve ========
void CondenseSolver::factor(const CondenseProblem& P, CondenseFactorization& F, const CondenseOptions& opt)
{
    check_dims(P);
    const int N =P.N, nx= P.nx, nu = P.nu;
    // 1) Γ_u, Γ_bb
    F.Gu.resize(N*nx, N*nu);
    F.Gbb.resize(N*nx);
    build_Gamma(P,F.Gu, F.Gbb);
    // 2) 块对角/堆叠
    F.Qbar.resize(N*nx, N*nx);  blkdiag_Qbar(P, F.Qbar);
    F.Sbar.resize(N*nu, N*nx);  blkdiag_Sbar(P, F.Sbar);
    F.Rbar.resize(N*nu, N*nu);  blkdiag_Rbar(P, F.Rbar);
    F.qbar.resize(N*nx); stack_qbar(P, F.qbar);
    F.sbar.resize(N*nu); stack_sbar(P, F.sbar);
    // 3) Hc = R̄ + Guᵀ Q̄ Gu + (Guᵀ S̄ + (Guᵀ S̄)ᵀ)
    F.Hc = F.Rbar;
    // tmp1 = Qbar * Gu
    MatrixXd tmp1 = F.Qbar * F.Gu; // (Nnx x Nnu)
    // Hc += Guᵀ * tmp1
    F.Hc.noalias() += F.Gu.transpose() * tmp1;
    // tmp2 = Guᵀ * Sbar
    MatrixXd tmp2 = F.Gu.transpose() * F.Sbar; // (Nnu x Nnu)
    if(opt.symmetrize_hessian) F.Hc.noalias() += tmp2 + tmp2.transpose();
    else F.Hc.noalias() += tmp2;
    // 4) gc = s̄ + Guᵀ (q̄ + Q̄ Γ_bb) + S̄ᵀ Γ_bb
    F.gc = F.sbar;
    VectorXd tmpv = F.qbar + F.Qbar * F.Gbb; // (Nnx)
    F.gc.noalias() += F.Gu.transpose() * tmpv;
    F.gc.noalias() += F.Sbar.transpose() * F.Gbb;
    // 5) Cholesky
    F.llt.emplace(F.Hc);
    if(F.llt->info()!=Eigen::Success){
        for(int i=0;i<F.Hc.rows();++i) F.Hc(i,i) += opt.diag_regularization;
        F.llt->compute(F.Hc);
        if(F.llt->info()!=Eigen::Success && opt.check_spd)
            throw std::runtime_error("Cholesky failed: Hc not SPD");
    }
}


void CondenseSolver::solve_with_factor(const CondenseProblem& P, const CondenseOptions& opt, CondenseFactorization& F, CondenseResult& res){
const int N=P.N, nx=P.nx, nu=P.nu;
// U* = -Hc^{-1} gc
VectorXd Ustar;
if (F.llt)
{
    Ustar = -F.llt->solve(F.gc);
}
else
{
    Ustar = -F.Hc.ldlt().solve(F.gc);
}
if(F.llt && F.llt->info()!=Eigen::Success) throw std::runtime_error("LLT solve failed");
// 回代
res.u.resize(N); res.x.resize(N+1);
res.x[0] = P.x0;
for(int n=0;n<N;++n){
res.u[n] = Ustar.segment(n*nu, nu);
const VectorXd bn = P.b.empty()? VectorXd::Zero(nx) : P.b[n];
res.x[n+1] = P.A[n]*res.x[n] + P.B[n]*res.u[n] + bn;
}
// 可选计算 cost
double J=0.0;
for(int n=0;n<N;++n){
const auto& x = res.x[n]; const auto& u = res.u[n];
const MatrixXd& Sn = P.S.empty()? MatrixXd::Zero(P.nu,P.nx) : P.S[n];
const VectorXd qn = P.q.empty()? VectorXd::Zero(nx) : P.q[n];
const VectorXd sn = P.s.empty()? VectorXd::Zero(nu) : P.s[n];
J += x.dot(P.Q[n]*x) + 2.0 * x.dot(Sn.transpose()*u) + u.dot(P.R[n]*u)
+ 2.0 * qn.dot(x) + 2.0 * sn.dot(u);
}
J += res.x[N].dot(P.QN*res.x[N]) + 2.0 * P.qN.dot(res.x[N]);
res.cost = J;
}


CondenseResult CondenseSolver::solve(const CondenseProblem& prob, const CondenseOptions& opt){
CondenseFactorization F; factor(prob, F, opt); CondenseResult R; solve_with_factor(prob, opt, F, R); return R;
}






}