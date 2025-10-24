#pragma once
#include <Eigen/Dense>
#include <vector>
#include <Eigen/Sparse>
#include <stdexcept>
#include <memory>

extern "C"
{
    #include <cblas.h>
    //#include <lapacke.h>
}

#define USE_QDLDL = 1

// 编译时 -DUSE_QDLDL打开
#ifdef USE_QDLDL
extern "C" {
    #include <qdldl.h>  // 只安装QDLDL库
}
#endif

/*
稠密MatrixXd 存储结构：连续二维数组 列主序
稀疏MatrixXd 三个一维数组(Ap/Ai/Ax), CSC
这里KKT是个大型稀疏矩阵，所以用稀疏MatrixXd
*/

/*
为什么要CscView和csc_view(),
把Eigen的稀疏矩阵(内部是CSC存储) “零拷贝“地暴露成QDLDL等外部稀疏求解器需要的三组原始指针(Ap/Ai/Ax)
*/

/*

QDLDL 的API不是 Eigen 类型,用的都是C风格CSC格式
Eigen 当然也用列主稀疏，但它把这些东西包在类里。外部库不认识 Eigen::SparseMatrix，只认 Ap/Ai/Ax 这三根指针

零拷贝、高性能
如果不用 CscView，你可能会写一段代码把 Eigen 的数据复制到 std::vector<int/double> 再传给求解器——这会浪费时间和内存。
CscView 直接借用（borrow）Eigen内部缓冲区的指针，不拷贝。

保证“压缩”状态
if (!A.isCompressed()) 的检查是为了确保 outerIndexPtr()/innerIndexPtr()/valuePtr() 指向的是连续的压缩存储（CSC）。不压缩时这些指针不安全。

类型对齐
外部库通常要求 int* 索引；Eigen 的索引类型可能是 64 位（ptrdiff_t）。
reinterpret_cast<const int*> 是在你明确索引不会溢出前提下的快速做法；更稳妥的做法是拷到 int 缓冲区再传（但那就不是零拷贝了）。

生命周期管理
因为是零拷贝，Ap/Ai/Ax 的生命期依赖于原始的 Eigen::SparseMatrix。
也就是说：只要外部库在用，你就不能让 Eigen 矩阵析构或触发realloc。

CscView 是指针视图：让 Eigen 稀疏矩阵直接被外部稀疏求解器使用；

解决了类型不匹配（Eigen类 vs C接口）和性能问题（避免复制）；

需要你确保：矩阵已压缩、索引位宽匹配或不溢出、生命周期安全。

只要你要把 Eigen 的稀疏矩阵丢给像 QDLDL 这样的底层库，这个小 struct 就特别好用；反之，如果你完全只用 Eigen 自带的分解器，它当然不是必须。

*/

/*
先用 Eigen 装配 KKT Sp K; K.setFromTriplets(...); K.makeCompressed();
调 auto V = csc_view(K); 得到 Ap/Ai/Ax
传给 QDLDL 做 symbolic（一次）+ numeric（每次）
update_numeric 时只改 K 的数值、makeCompressed() 后再把同样的 Ap/Ai/Ax（同pattern）喂给 numeric
*/

// QDLDLData 最重要的作用：保存结构 + 重复利用

using Sp = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using T = Eigen::Triplet<double>;
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;


struct LQRProblem
{
    // 维度
    int nx = 0;
    int nu = 0;
    int N = 0;

    // LQR的参数，稠密
    std::vector<Eigen::MatrixXd> A;
    std::vector<Eigen::MatrixXd> B;
    std::vector<Eigen::MatrixXd> Q;
    std::vector<Eigen::MatrixXd> R;

    std::vector<Eigen::MatrixXd> S; 
    std::vector<Eigen::VectorXd> q;
    std::vector<Eigen::VectorXd> s;
    std::vector<Eigen::VectorXd> b;

    Eigen::MatrixXd P_N;
    Eigen::VectorXd p_N;

    Eigen::VectorXd x0;
};

struct LQRResult
{
    // 反馈
    std::vector<Eigen::MatrixXd> K;
    std::vector<Eigen::VectorXd> k;

    std::vector<Eigen::MatrixXd> P;
    std::vector<Eigen::VectorXd> p;

    std::vector<Eigen::VectorXd> x;     // N+1
    std::vector<Eigen::VectorXd> u;     // N

};

// QDLDL需要CSC视图 零拷贝
struct Cscview {
    int n;
    const int* Ap;  // n+1
    const int* Ai;  // nnz
    const double* Ax;// nnz
};

Cscview csc_view(const Sp& A);

// 为了调用qdldl库，需要稀疏矩阵，如果稠密矩阵，直接用Eigen::LDLT或者LAPACKE_dpotrf
// KKT矩阵一般是稀疏的

class QDLDLSolver
{
public:
    struct Options
    {
        double sigma = 1e-9;    // 上左块正则(H+sigma I)
        double rho = 1e-9;      // 右下块正则 （-rho I)
        bool use_amd = true;    // 对称AMD重排
        bool verbose = false;   // 打印信息
    };

    QDLDLSolver() = default;

    // 一次性构建 (结构固定时: N nx nu 不变)
    void initialize(const LQRProblem& prob, const Options& opt = Options{});

    // 只数值更新
    void update_numeric(const LQRProblem& prob);

    // 求解KKT: 得到[z; lambda], 并拆成result.x result.u
    void solve(LQRResult& result);
    void solve_riccati(const LQRProblem& prob, LQRResult& res, double reg = 1e-9) const;

    // 调试
    const Sp& H() const {return H_;}
    const Sp& A() const {return A_;}
    const Vec& g() const {return g_;}
    const Vec& h() const {return h_;}
    const Sp& K_upper() const {return K_upper_;}

private:
    // 变量偏移
    inline int off_x_(int k) const {return k*nx_; }
    inline int off_u_(int k) const {return (N_+1)*nx_ + k*nu_; }

    // 尺寸检查
    void check_optional_sizes_ (const LQRProblem& p) const;

    // 装配稀疏QP
    void form_P_Hg_(const LQRProblem& prob, bool numeric_only = false); // 构 H

    void form_Ah_(const LQRProblem& prob, bool numeric_only = false);
    // 组上三角 KKT & 置换 (KKT用)
    void form_K_upper_(bool numeric_only = false);
    void apply_symmetric_amd_();
    void factorize_();
    void refactorize_numeric_();
    void solve_linear_(const Vec& rhs_perm, Vec& x_perm);

    // rhs 打包/解包(KKT 用)
    void pack_rhs_(Vec& rhs) const;
    void unpack_solution_(const Vec& sol, LQRResult& res) const;


private:
    // 尺寸
    int nx_ = 0;
    int nu_ = 0;
    int N_  = 0;
    int nvar_ = 0;
    int ncon_ = 0;
    int nK_ = 0;

    Options opt_;

    // 稀疏QP(KKT用)
    Sp H_;  // nvar x nvar
    Sp A_;  // ncon x nvar
    Vec g_; // nvar
    Vec h_; // ncon

    // 上三角 KKT 及其置换
    Sp K_upper_;    // (nvar + ncon) x (nvar + ncon) 仅上三角
    Sp Kp_upper_;   // 置换后的上三角
    Eigen::PermutationMatrix<Eigen::Dynamic> P_;
    bool have_perm_ = false;

    #ifdef USE_QDLDL
        // QDLDL
        struct QDLDLData{
            QDLDL_int Ln = 0, sumLnz = 0;
            std::vector<QDLDL_int> Lp, Li, etree, Lnz, iwork;
            std::vector<QDLDL_bool> bwork;
            std::vector<QDLDL_float> Lx, D, Dinv, fwork, x; // x 作为解的缓冲
            bool symbolic_ready = false;
            bool numeric_ready = false;
        } F_;

    #else
        // 回退 Eigen稀疏 LU (仅KKT路线需要)
        Eigen::SparseLU<Sp> lu_;
        bool symbolic_ready_ = false;
        bool numeric_ready_ = false;
    #endif
};






