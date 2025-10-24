#include "schur_blas.hpp"
#include <Eigen/Dense>
#include <cblas.h>
#include <lapacke.h>
#include <stdexcept>
#include <cassert>
#include <string>

// BLAS = 基础线性代数运算(矩阵乘法，三角解，向量点积)
// LAPACK = 高层算法(LU/QR/SVD/特征值分解)，依赖BLAS
// LAPACK = C/C++的入口
// Eigen/Dense = 稠密矩阵/向量

namespace ocp::schur_blas {

// 上三角Cholesky分解 (上三角)
// A 对称正定 -> U 满足 A = U^T U
// 结果就地写回到A 的上三角 下三角未定义
inline void chol_upper(Eigen::MatrixXd& A)
{
    assert(A.rows() == A.cols());
    const int n = (int)A.rows();
    // dpotrf "U"表示只写/读上三角
    // A.data() 必须是列主序的连续内存 lda = n
    int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', n, A.data(), n);
    if (infor != 0) throw std::runtime_error("dpotrf failed (not SPD)");
}  

// 上三角矩阵求逆(就地)
// U := inv(U),要求U非奇异的上三角
inline void inv_tri_upper(Eigen::MatrixXd& U)
{   
    // 输入检查
    assert(U.rows() == U.cols());
    const int n = (int)U.rows();
    // dtrtri "U" 上三角， N - 非单位对角
    int info = LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', n, U.data(), n);
    if (info != 0) throw std::runtime_error("dtrtri failed (singular U)");
}

// 右乘三角矩阵 B = B * U
// 其中 U为上三角(非单位) B任意mxn U 为 nxn
inline void trmm_right_upper_noTrans(Eigen::MatrixXd& B, const Eigen::MatrixXd& U)
{   
    assert(U.rows() == U.cols());
    assert(B.cols() == U.rows());

    const int m = (int)B.rows();
    const int n = (int)B.cols();
    // 
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                m, n, 1.0, U.data(), (int)U.rows(), B.data(), (int)B.rows());

}

// 对称秩 k 更新 (Upper 存储)
// C = alpha *A * A^T + beta *C
// 要求C为nxn对称， 且只看/维护其上三角
// A 尺寸 nxk
inline void syrk_upper_nn(Eigen::MatrixXd& C, const Eigen::MatrixXd& A, double alpha = 1.0, double beta = 1.0)
{
    assert(C.rows() == C.cols());
    assert(C.rows() == C.cols());

    const int n = (int)C.rows();
    const int k = (int)A.cols();
    // dsyrk: Uplo=Upper；Trans=NoTrans（用 A*A^T）
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                n, k, alpha, A.data(), (int)A.rows(), beta, C.data(), (int)C.rows());

}

// U = U * U^T (Upper分支)
// DLAUUM 把上三角 U 就地改写为 U*U^T（仍仅写上三角）
// 常用于：先对 U 做三角逆，再 lauum 得到 (U^{-1})(U^{-1})^T = (U^T U)^{-1}
inline void lauum_upper(Eigen::MatrixXd& U)
{
    int n = (int)U.rows();
    int info = LAPACKE_dlauum(LAPACK_COL_MAJOR, 'U', n, U.data(), n);
    if (info != 0) throw std::runtime_error("dlauum failed");
}

// 一般矩阵乘 C = alpha * A * B + beta *C
// A mxk, B kxn, C mxn
inline void gemm_nn(Eigen::MatrixXd& C, 
                    const Eigen::MatrixXd& A,
                    const Eigen::MatrixXd& B, double alpha = 1.0, double beta = 1.0)
{
    assert(A.rows() ==  C.rows());
    assert(B.cols() ==  C.cols());
    assert(A.cols() ==  B.rows());
    const int m = (int)C.rows();
    const int n = (int)C.cols();
    const int k = (int)A.cols();

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, 
                A.data(), (int)A.rows(), 
                B.data(), (int)B.rows(), 
                beta, 
                C.data(), (int)C.rows());
}

// 解上三角转置系统 U^T x = b U为上三角(非单位矩阵) b为n维
// solve U^T x = b (U upper -> U^T lower)
inline Eigen::VectorXd solve_UT_x(const Eigen::MatrixXd& U, const Eigen::VectorXd& b)
{
    assert(U.rows() == U.cols());
    assert(U.rows() == b.size());
    Eigen::VectorXd x = b;
    // dtrsv: UpLo=Upper；Trans=Trans（因为是 U^T x = b）
    cblas_dtrsv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit, (int)U.rows(), U.data(), 
    (int)U.rows(), x.data(), 1);
    return x;
}

// solve U x = b (U upper)
inline Eigen::VectorXd solve_U_x(const Eigen::MatrixXd& U, const Eigen::VectorXd& b)
{
    assert(U.rows() == U.cols());
    assert(U.rows() == b.size());
    Eigen::VectorXd x = b;
    // dtrsv: UpLo=Upper；Trans=NoTrans（U x = b）
    cblas_dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, (int)U.rows(), U.data(),
    (int)U.rows(), x.data(), 1);
    return x;
}

// 右侧三角解 X = M * U^{-1}
// X =  M * U^{-1}
// 等价于解 X * U = M M:mxn U:nxn (上三角)
inline Eigen::MatrixXd right_mult_Uinv(const Eigen::MatrixXd& M, const Eigen::MatrixXd& U)
{
    assert(U.rows() == U.cols());
    assert(M.cols() == U.rows());
    Eigen::MatrixXd X = M;  // 结果覆盖在X上
    // dtrsm: Side=Right → 解 X * op(U) = M
    //        Uplo=Upper；TransA=NoTrans；Diag=NonUnit；α=1
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
    (int)X.rows(), (int)X.cols(), 1.0, 
    U.data(), (int)U.rows(), 
    X.data(), (int)X.rows());
    return X;

}


struct Inputs {
    int N, nx, nu;
    std::vector<Eigen::MatrixXd> Q, S, R, A, B;
    std::vector<Eigen::VectorXd> q, s, b;
    Eigen::MatrixXd QN;
    Eigen::VectorXd qN;
    Eigen::VectorXd x0;
}

Outputs solve(const Inputs& I)
{
    const int N = I.N, nx = I.nx, nu = I.nu;
    
    // 对H进行Cholesky分解, 得到U，U是上三角矩阵
    Eigen::MatrixXd U0_22 = I.R[0];
    chol_upper(U0_22);

    // UFv是H的cholesky的上三角分解矩阵
    struct UF {Eigen::MatrixXd U11, U12, U22;};
    std::vector<UF> UFv(N);
    for (int n = 1; n <= N-1; ++n)
    {
        Eigen::MatrixXd H(nx+nu, nx+nu);
        H.setZero();
        
        H.topLeftCorner(nx, nx) = I.Q[n];
        H.topRightCorner(nx, nu) = I.S[n].transpose();
        H.bottomLeftCorner(nu, nx) = I.S[n];
        H.bottomRightCorner(nu, nu) = I.R[n];

        chol_upper(H);
        
        UFv[n].U11 = H.topLeftCorner(nx, nx);
        UFv[n].U12 = H.topRightCorner(nx, nu);
        UFv[n].U22 = H.bottomRightCorner(nu, nu);
    }

    Eigen::MatrixXd UN_11 = I.QN;
    chol_upper(UN_11);


    // 求逆
    Eigen::MatrixXd U0_22_inv = U0_22;
    inv_tri_upper(U0_22_inv);

    std::vector<UF> Uinv(N);
    for (int n = 1; n <= N-1; ++n)
    {
        Uinv[n].U11 = UFv[n].U11;
        inv_tri_upper(Uinv[n].U11);
        Uinv[n].U22 = UFv[n].U22;
        inv_tri_upper(Uinv[n].U22)
        Uinv[n].U12 = -Uinv[n].U11 * UFv[n].U12 * Uinv[n].U22;

    }
    Eigen::MatrixXd UN_11_inv = UN_11;
    inv_tri_upper(UN_11_inv);

    // 计算Phi，这个Phi不是方针，也不是上三角矩阵。
    Eigen::MatrixXd Phi0_22 = -I.B[0] * Uinv[0].U22;
    Phi0_22 = - I.B[0] * U0_22_inv;

    std::vector<Eigen::MatrixXd> Phi11(N), Phi12(N), Phi21(N), Phi22(N);
    for (int n = 1; n<=N-1; ++n)
    {
        Phi11[n] = Uinv[n].U11;
        Phi12[n] = Uinv[n].U12;
        Phi21[n] = -I.A[n] * Uinv[n].U11;
        Phi22[n] = -I.B[n] * Uinv[n].U22;
    }
    Eigen::MatrixXd PhiN_11 = UN_11_inv;

    // 计算Schur补矩阵 
    // PsiDiag[k]: 第k个对角块Psi_(k,k)，大小为nx nx
    // PsiDiag[k]: 第k个上超对叫块Psi_(k, k+1)，大小为nx nx
    // Phi11[n]: 
    std::vector<Eigen::MatrixXd> PsiDiag(N+1), PsiOff(N);
    PsiDiag[1] = Eigen::MatrixXd::Zero(nx, nx);
    syrk_upper_nn(PsiDiag[1], Phi0_22);
    syrk_upper_nn(PsiDiag[1], Phi12[1], 1.0, 1.0);
    {
        Eigen::MatrixXd tmp = Phi11[1];
        lauum_upper(tmp);
        PsiDiag[1].triangularView<Eigen::Upper>() += tmp.triangularView<Eigen::Upper<();
    }
    PsiDiag[1] = 0.5*(PsiDiag[1] + PsiDiag[1].transpose());

    PsiOff[1] = Eigen::MatrixXd::Zero(nx, nx);
    PsiOff[1] += Phi11[1] * Phi21[1].transpose();
    PsiOff[1] += Phi22[1] * Phi21[1].transpose();

    for (int n = 2; n <= N-1; ++n)
    {   
        // 第n个
        PsiDiag[n] = Eigen::MatrixXd::Zero(nx, nx);
        syrk_upper_nn(PsiDiag[n], Phi21[n-1]);      // + Phi_{n-1,21} * Phi_{n-1, 21}^T
        syrk_upper_nn(PsiDiag[n], Phi22[n-1], 1.0, 1.0);    // + Phi_{n-1, 22} * Phi_{n-1, 22}^T
        Eigen::MatrixXd tmp = Phi11[n];
        lauum_upper(tmp);       // tmp = Phi_{n,11} * Phi_{n,11}^T
        PsiDiag[n].triangularView<Eigen::Upper>() += 
            tmp.triangularView<Eigen::Upper>();
        syrk_upper_nn(PsiDiag[n], Psi12[n], 1.0, 1.0);  // Phi_{n, 12} * Phi_{n, 12}^T
        PsiDiag[n] = 0.5 * (PsiDiag[n]+PsiDiag[n].transpose());  
        

        // 第n个
        PsiOff[n] = Phi11[n] * Phi21[n].transpose() + Phi12[n] * Phi22[n].transpose();
    }

    // 第N个方阵
    PsiDiag[N] = Mat::Zero(nx,nx);
    syrk_upper_nn(PsiDiag[N], Phi21[N-1]);
    syrk_upper_nn(PsiDiag[N], Phi22[N-1], 1.0, 1.0);
    { 
        Eigen::MatrixXd tmp = PhiN_11; lauum_upper(tmp);
        PsiDiag[N].triangularView<Eigen::Upper>() += tmp.triangularView<Eigen::Upper>(); 
    }
    PsiDiag[N] = 0.5*(PsiDiag[N]+PsiDiag[N].transpose());

    // 对Schur补矩阵进行cholesky分解，得到上三角矩阵
    std::vector<Eigen::MatrixXd> UpsiDiag(N+1), UpsiOff(N);
    UpsiDiag[1] = PsiDiag[1];
    chol_upper(UpsiDiag[1]);
    for (int n = 2; n <= N; ++n)
    {
        //U_{n-1, n} = U_{n-1, n-1}^{T} * Psi_{n-1, n}
        UpsiOff[n-1] = PsiOff[n-1];
        // solve U^T X = PsiOff -> X
        clbas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
                    nx, nx, 1.0, UpsiDiag[n-1].data(), nx, UpsiOff[n-1].data(), nx);
        // Schur diag
        Eigen::MatrixXd S = PsiDiag[n] - UpsiOff[n-1].transpose() * UpsiOff[n-1];
        UpsiDiag[n] = S;
        chol_upper(UpsiDiag[n]);
    }
    
    // --- 5) φ vectors (dtrmv) and β (dgemv) ---
    Eigen::VectorXd phi0_2 = solve_UT_x(U0_22, I.s[0] + I.S[0]*I.x0); // U^T w = rhs
    std::vector<Eigen::VectorXd> phi1(N), phi2(N);
    for(int n=1;n<=N-1;++n)
    {
        // Using explicit U^{-T} through inv triangulars: [phi1;phi2] = U^{-T} [q;s]
        // Solve by two triangular solves would avoid forming inverse; we precomputed Uinv for clarity.
        Eigen::VectorXd rhs(nx+nu); rhs << I.q[n], I.s[n];
        // w = U^{-T} rhs = (U^{-1})^T rhs, but we have Uinv; so split by blocks:
        phi1[n] = Uinv[n].U11.transpose() * I.q[n] + Uinv[n].U12.transpose() * I.s[n];
        phi2[n] = Uinv[n].U22.transpose() * I.s[n];
    }
    Eigen::VectorXd phiN_1 = UN_11_inv.transpose() * I.qN;


    std::vector<Eigen::VectorXd> beta(N+1);
    beta[1] = I.b[0] + I.A[0]*I.x0 + Phi0_22*phi0_2 + Phi11[1]*phi1[1] + Phi12[1]*phi2[1];
    for(int n=2;n<=N-1;++n)
    {
        beta[n] = I.b[n-1] + Phi21[n-1]*phi1[n-1] + Phi22[n-1]*phi2[n-1]
        + Phi11[n]*phi1[n] + Phi12[n]*phi2[n];
    }
    beta[N] = I.b[N-1] + Phi21[N-1]*phi1[N-1] + Phi22[N-1]*phi2[N-1] + PhiN_11*phiN_1;

    // 计算gamma 和pi
    std::vector<Eigen::VectorXd> gamma(N+1), pi(N+1);
    gamma[1] = solve_UT_x(UpsiDiag[1], beta[1]);
    for (int n=2; n<=N; ++n)
    {
        Eigen::VectorXd rhs = beta[n] - UpsiOff[n-1].transpose() * gamma[n-1];
        gamma[n] = solve_UT_x(UpsiDiag[n], rhs);    // 这里注意，UpsiDiag没有做转置
    }
    pi[N] = solve_UT_x(UpsiDiag[N], gamma[N]);
    for (int n= N-1; n>=1; --n)
    {
        Eigen::VectorXd rhs = gamma[n] - UpsiOff[n]*pi[n+1];
        pi[n] = solve_UT_x(UpsiDiag[n], rhs);
    }   

    // 计算x和u
    Outputs out;
    out.x.resize(N+1);
    out.u.resize(N);
    out.x[0] = I.x0;

    // u0 = U0^{-1} U0^{T} rhs
    {
        Eigen::VectorXd rhs = -I.B[0].transpose() * pi[1] - (I.s[0] + I.S[0] * I.x0);
        Eigen::VectorXd w = solve_UT_x(U0_22, rhs);
        out.u[0] = solve_UT_x(U0_22, w);
    }
    for (int n = 1; n<= N-1; ++n)
    {
        // Build U block back ( or reuse UFv[n])
        Eigen::MatrixXd U(nx+nu, nx+nu);
        U.setZero();
        // 左上块 U11 上三角
        U.topLeftCorner(nx, nx) = UFv[n].U11;
        // 右上块 U12 任意，但是保证整体U上三角
        U.topRightCorner(nx, nu) = UFv[n].U12;
        // 右下块 U22 上三角
        U.bottomRightCorner(nu, nu) = UFv[n].U22;

        // 构造右端rhs
        // 上半部 (对应x_n的方程)
        // rhs_x = pi_n - A_n^T pi_{n+1} - q_n
        Eigen::VectorXd rhs(nx + nu);
        rhs.head(nx) = pi[n] - I.A[n].transpose() * pi[n+1] - I.q[n];

        // 下半部 (对应u_n的方程)
        // rhs_u = -B_n^T pi_{n+1} - s_n
        rhs.tail(nu) = -I.B[n].transpose() * pi[n+1] - I.s[n];

        // 用分解后的U做 两步三角求解 相当于解(U^T U) y = rhs
        // 先解 U^T w = rhs (回代)
        Eigen::VectorXd w = solve_UT_x(U, rhs);
        // 再解 U * y = w (前代)
        Eigen::VectorXd y = solve_UT_x(U, w);

        // 取出解向量的前后块，得到本阶段的x_n 和 u_n
        out.x[n] = y.head(nx);  // y的前nx个分量 x_n
        out.u[n] = y.tail(nu);  // y的后nu个分量 u_n
    }

    // 末端阶段 n = N 通常只有x_N 无u_N, 因子就是UN_11 上三角
    {
        // rhs_N = pi_N - q_N
        Eigen::VectorXd rhs = pi[N] - I.qN;

        // 同样两步三角求解 解 (UN_11^T, UN_11) x_N = rhs
        Eigen::VectorXd w = solve_UT_x(UN_11, rhs);
        out.x[N] = solve_UT_x(UN_11, w);
    }

    return out;
}   






}