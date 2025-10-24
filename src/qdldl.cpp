#include "qdldl.hpp"
#include <iostream>

// CSC视图
Cscview csc_view(const Sp& A)
{
    if (!A.isCompressed())
        throw std::runtime_error("Eigen::SparseMatrix must be compressed (call .makeCompressed())");
    Cscview v;
    v.n = static_cast<int>(A.cols());
    v.Ap = reinterpret_cast<const int*>(A.outerIndexPtr());
    v.Ai = reinterpret_cast<const int*>(A.innerIndexPtr());
    v.Ax = A.valuePtr();
    return v;
}

void QDLDLSolver::check_optional_sizes_(const LQRProblem& p) const {
    auto chk = [&](const auto& v, const char* name)
    {
        if (!v.empty() && (int)v.size()!= N_ )
            throw std::runtime_error(std::string(name)+"size must be N or empty");
    };
    chk(p.S, "S"); chk(p.q, "q"); chk(p.s, "s");  chk(p.b, "b");
}

void QDLDLSolver::initialize(const LQRProblem& prob, const Options& opt)
{
    opt_ = opt;
    nx_ = prob.nx; 
    nu_ = prob.nu;
    N_ = prob.N;
    if (nx_ < 0 || N_ < 0) throw std::runtime_error("Invalud nx/N");
    if (N_>0 && ((int)prob.A.size()!=N_ || (int)prob.B.size()!=N_
            || (int)prob.Q.size()!=N_ || (int)prob.R.size()!=N_))
        throw std::runtime_error("A/B/Q/R size must be N");
        check_optional_sizes_(prob);

    nvar_ = (N_ + 1)*nx_ + N_*nu_;
    ncon_ = N_ * nx_;
    nK_ = nvar_ + ncon_;

    form_P_Hg_(prob, false);
    form_Ah_(prob, false);
    form_K_upper_(false);
    if (opt_.use_amd) apply_symmetric_amd_();
    else {Kp_upper_ = K_upper_; have_perm_ = false;}
    factorize_();
}

// 数值更新kkt
void QDLDLSolver::update_numeric(const LQRProblem& prob)
{
    if (prob.nx!=nx_ || prob.nu!=nu_ || prob.N != N_)
        throw std::runtime_error("Structure changed; call initialize() again");
    form_P_Hg_(prob, true);
    form_Ah_(prob, true);
    form_K_upper_(true);

    if(have_perm_)
    {
        Kp_upper_ = P_ * K_upper_ * P_.transpose();
        Kp_upper_.makeCompressed();
    }
    else
    {
        Kp_upper_ = K_upper_;
    }
    refactorize_numeric_();
}

// KKT求解
void QDLDLSolver::solve(LQRResult& result)
{
    result.x.assign(N_+1, Vec::Zero(nx_));
    result.u.assign(N_, Vec::Zero(nu_));

    Vec rhs(nK_);   pack_rhs_(rhs);
    Vec rhs_p = have_perm_ ? (P_ * rhs) : rhs;

    Vec sol_p(nK_);
    solve_linear_(rhs_p, sol_p);

    Vec sol = have_perm_ ? (P_.transpose() * sol_p) : sol_p;
    unpack_solution_(sol, result);
}

// 装 H,g (KKT)
void QDLDLSolver::form_P_Hg_(const LQRProblem& p, bool numeric_only)
{
    if (!numeric_only) {H_.resize(nvar_, nvar_); H_.setZero(); g_.setZero();}
    else {H_.setZero(); g_.setZero();}
    std::vector<T> tri;
    tri.reserve((size_t)N_*(nx_*nx_ + nu_*nu_) + (size_t)N_* 2*nx_*nu_ + nx_*nx_); 

    auto add_sym_block_upper = [&](int roff, const Mat& M)
    {
        if (M.rows() != M.cols()) throw std::runtime_error("add_sym_block_upper: square needed");
        int n = (int)M.rows();
        for (int j=0; j <n; ++j)
        {
            for (int i=0; i<=j; ++i)
            {
                double v = M(i,j);
                if (v!=0.0) tri.emplace_back(roff+i, roff+j, v);
            }
            
        }
    };

    auto add_rect_block = [&](int roff, int coff, const Mat& M)
    {
        for (int j = 0; j <M.cols(); ++j)
        {
            for (int i = 0; i <M.rows(); ++i)
            {
                double v = M(i,j);
                if (v!=0.0) tri.emplace_back(roff+i, coff+j, v);
            }
        }
    };

    g_.setZero(nvar_);

    for (int k = 0; k<N_; ++k)
    {
        if (!p.Q.empty()) add_sym_block_upper(off_x_(k), p.Q[k]);
        if (!p.R.empty()) add_sym_block_upper(off_u_(k), p.R[k]);
        if (!p.s.empty() && p.S[k].rows() == nu_ && p.S[k].cols() == nx_)
        {
            // 上三角写(x,u)的S^T, 对称的(u,x)不需重复写
            add_rect_block(off_x_(k), off_u_(k), p.S[k].transpose());
        }
        if (p.P_N.size() == nx_*nx_) add_sym_block_upper(off_x_(N_), p.P_N);
        if (p.P_N.size() == nx_)    g_.segment(off_x_(N_), nx_) += p.P_N;
        
        H_.setFromTriplets(tri.begin(), tri.end());
        H_.makeCompressed();
    
    }
}


// 装A, h
void QDLDLSolver::form_Ah_(const LQRProblem& p, bool numeric_only)
{
    if (!numeric_only) { A_.resize(ncon_, nvar_); A_.setZero(); h_.setZero(ncon_); }
    else { A_.setZero(); h_.setZero(); }

    std::vector<T> tri;
    tri.reserve((size_t)N_*(nx_*nx_ + nx_*nu_ + nx_));

    int row = 0;
    for (int k=0;k<N_;++k) {
        for (int i=0;i<nx_;++i) tri.emplace_back(row+i, off_x_(k+1)+i, 1.0);
        if (!p.A.empty()) {
        for (int j=0;j<nx_;++j)
            for (int i=0;i<nx_;++i) { double v=p.A[k](i,j); if(v!=0.0) tri.emplace_back(row+i, off_x_(k)+j, -v); }
        }
        if (!p.B.empty()) {
            for (int j=0;j<nu_;++j)
            for (int i=0;i<nx_;++i) { double v=p.B[k](i,j); if(v!=0.0) tri.emplace_back(row+i, off_u_(k)+j, -v); }
        }
        if (!p.b.empty() && p.b[k].size()==nx_) h_.segment(row, nx_) = p.b[k];
        row += nx_;
    }

    A_.setFromTriplets(tri.begin(), tri.end());
    A_.makeCompressed();
}

// 组上三角KKT
void QDLDLSolver::form_K_upper_(bool numeric_only)
{
    if (!numeric_only) {K_upper_.resize(nK_, nK_); K_upper_.setZero();}
    else {K_upper_.setZero();}

    std::vector<T> tri;
    tri.reserve((size_t)(H_.nonZeros() + A_.nonZeros() + nvar_ + ncon_));

    // 左上: H+sigma*I(仅上三角已有)
    for (int k=0;k<H_.outerSize();++k)
        for (Sp::InnerIterator it(H_,k); it; ++it)
            if (it.row() <= it.col()) tri.emplace_back(it.row(), it.col(), it.value());
    for (int i=0;i<nvar_;++i) tri.emplace_back(i,i,opt_.sigma);

    // 右上：Aᵀ
    for (int col=0; col<A_.outerSize(); ++col)
        for (Sp::InnerIterator it(A_, col); it; ++it) {
      int r = it.row(), c = it.col();
      tri.emplace_back(c, nvar_ + r, it.value());
    }

    // 右下：-rho*I（仅对角）
    for (int i=0;i<ncon_;++i) tri.emplace_back(nvar_+i, nvar_+i, -opt_.rho);

    K_upper_.setFromTriplets(tri.begin(), tri.end());
    K_upper_.makeCompressed();
}

// AMD置换 KKT
void QDLDLSolver::apply_symmetric_amd_()
{
    Eigen::AMDOrdering<int> amd;
    Eigen::PermutationMatrix<Eigen::Dynamic> P;
    amd(K_upper_, P);
    P_ = P;
    have_perm_ = true;
    Kp_upper_  = P_ * K_upper_ * P_.transpose();
    Kp_upper_.makeCompressed();
}

// ============ 打包/解包（KKT） ============
void QDLDLSolver::pack_rhs_(Vec& rhs) const {
  rhs.setZero(nK_);
  rhs.head(nvar_) = -g_;
  rhs.tail(ncon_) = -h_;
}
void QDLDLSolver::unpack_solution_(const Vec& sol, LQRResult& res) const {
  const Vec z = sol.head(nvar_);
  for (int k=0;k<=N_;++k) res.x[k] = z.segment(off_x_(k), nx_);
  for (int k=0;k<N_; ++k) res.u[k] = z.segment(off_u_(k), nu_);
}

// 分解KKT
void QDLDLSolver::factorize_()
{
    #ifdef USE_QDLDL
        auto V = csc_view(Kp_upper_);
        F_.Ln = (QDLDL_int)V.n;
        F_.etree.assign(F_.Ln, 0);
        F_.Lnz.assign(F_.Ln, 0);
        QDLDL_etree(F_.Ln, 
                    reinterpret_cast<const QDLDL_int*>(V.Ap),
                    reinterpret_cast<const QDLDL_int*>(V.Ai),
                    F_.etree.data(), F_.Lnz.data());
        F_.sumLnz = 0; for (QDLDL_int j = 0; j<F_.Ln; ++j) F_.sumLnz += F_.Lnz[j];
        F_.Lp.assign(F_.Ln+1, 0);
        for(QDLDL_int j = 0; j <F_.Ln; ++j) F_.Lp[j+1] = F_.Lp[j] + F_.Lnz[j];
        F_.Li.assign(F_.sumLnz, 0);
        F_.Lx.assign(F_.sumLnz, 0.0);
        F_.D.assign(F_.Ln, 0.0);
        F_.Dinv.assign(F_.Ln, 0.0);
        F_.iwork.assign(3*F_.Ln, 0);
        F_.bwork.assign(F_.Ln, 0);
        F_.fwork.assign(F_.Ln, 0.0);
        F_.x.assign(F_.Ln, 0.0);
        
        QDLDL_factor(F_.Ln, 
                    reinterpret_cast<const QDLDL_int*>(V.Ap),
                    reinterpret_cast<const QDLDL_int*>(V.Ai),
                    reinterpret_cast<const QDLDL_float*>(V.Ax),
                    F_.Lp.data(), F_.Li.data(), F_.Lx.data(),
                    F_.D.data(), F_.Dinv.data(),
                    F_.etree.data(), F_.Lnz.data(),
                    F_.iwork.data(), F_.bwork.data(), F_.fwork.data());
        F_.symbolic_ready = true;
        F_.numeric_ready = true;
        if (opt_.verbose) std::cout << "[QDLDL] factorized (n="<<F_.Ln<<", nnz(L)="<<F_.sumLnz<<")\n";
    #else
        if (opt_.verbose) std::cout << "[WARN] USE_QDLDL not defined, using SparseLU fallback\n";
        lu_.analyzePattern(Kp_upper_);
        lu_.factorize(Kp_upper_);
        if (lu_.info()!=Eigen::Success) throw std::runtime_error("SparseLU factorize failed");
        symbolic_ready_ = true;
        numeric_ready_  = true;
    #endif
}

// 数值重分解KKT
void QDLDLSolver::refactorize_numeric_() {
#ifdef USE_QDLDL
  if (!F_.symbolic_ready) throw std::runtime_error("QDLDL symbolic not ready");
  auto V = csc_view(Kp_upper_);
  QDLDL_factor(F_.Ln,
               reinterpret_cast<const QDLDL_int*>(V.Ap),
               reinterpret_cast<const QDLDL_int*>(V.Ai),
               reinterpret_cast<const QDLDL_float*>(V.Ax),
               F_.Lp.data(), F_.Li.data(), F_.Lx.data(),
               F_.D.data(), F_.Dinv.data(),
               F_.etree.data(), F_.Lnz.data(),
               F_.iwork.data(), F_.bwork.data(), F_.fwork.data());
  F_.numeric_ready = true;
#else
  lu_.factorize(Kp_upper_);
  if (lu_.info()!=Eigen::Success) throw std::runtime_error("SparseLU refactorize failed");
  numeric_ready_ = true;
#endif
}


// 线性求解KKT
void QDLDLSolver::solve_linear_(const Vec& rhs_perm, Vec& x_perm)
{
    #ifdef USE_QDLDL
        if (!F_.numeric_ready) throw std::runtime_error("QDLDL numeric not ready");
        if ((int)rhs_perm.size()!=F_.Ln) throw std::runtime_error("rhs size mismatch");
        for (int i = 0; i< F_.Ln; ++i) F_.x[i] = (QDLDL_float)rhs_perm[i];
        QDLDL_solve(F_.Ln, F_.Lp.data(), F_.Li.data(), F_.Lx.data(), F_.Dinv.data(), F_.x.data());
        x_perm.resize(F_.Ln);
        for (int i =0; i<F_.Ln; ++i) x_perm[i] = (double)F_.x[i];
    #else
        if (!numeric_ready_) throw std::runtime_error("SparseLU numeric not ready");
        x_perm = lu_.solve(rhs_perm);
        if (lu_.info() != Eigen::Success) throw std::runtime_error("SparseLU solve failed");
    #endif
}

