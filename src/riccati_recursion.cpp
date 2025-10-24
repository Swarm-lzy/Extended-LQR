#include "riccati_recursion.hpp"
#include <cstring>  //

namespace mpc
{
    // 主算法
    RiccatiResult RiccatiSolver::solve(const RiccatiProblem& prob, const RiccatiOptions& opt)
    {
        const int nx = prob.nx;
        const int nu = prob.nu;
        const int N = prob.N;

        // 基本尺寸检查
        assert_size(nx>0 && nu>0 && N>0, "nx, nu, N must be positive.");
        assert_size((int)prob.A.size()==N, "A size must be N.");
        assert_size((int)prob.B.size()==N, "B size must be N.");
        assert_size((int)prob.Q.size()==N, "Q size must be N.");
        assert_size((int)prob.R.size()==N, "R size must be N.");
        
        // 终端项检查
        assert_size(prob.P_N.rows()==nx && prob.P_N.cols()==nx, "P_N dim mismatch.");
        assert_size(prob.p_N.size()==nx, "p_N dim mismatch.");

        // 可选量允许缺失：若某 n 缺失则当作 0 处理
        auto hasS = ((int)prob.S.size()==N);
        auto hasq = ((int)prob.q.size()==N);
        auto hass = ((int)prob.s.size()==N);
        auto hasb = ((int)prob.b.size()==N);
        
        // 结果容器
        RiccatiResult res;
        res.K.resize(N);
        res.k.resize(N);
        res.P.resize(N+1);
        res.p.resize(N+1);

        // 后向递推工作区
        Eigen::MatrixXd Pn1 = prob.P_N; // P_{n+1}
        Eigen::VectorXd pn1 = prob.p_N; // p_{n+1}

        // 为计算中间项准备缓存(避免频繁分配)
        Eigen::MatrixXd Re(nu, nu); // R_e = R + B' P B
        Eigen::MatrixXd PA(nx, nx); // P_{n+1} * A_n
        Eigen::MatrixXd PB(nx, nu); // P_{n+1} * B_n
        Eigen::MatrixXd BtPA(nu, nx);   // B' * (P A)
        Eigen::MatrixXd L(nu,nx);   // Λ^{-1} * (S + B' P A)
        Eigen::VectorXd t(nx);  // t = P_{n+1} b + p_{n+1}
        Eigen::VectorXd se(nu);     // s_e = s + B' t
        Eigen::MatrixXd tmp(nx, nx);    // 临时用于A' P A / L' L等

        // 后向 n = N-1, ... 0
        for (int n = N-1; n>=0; --n)
        {
            const auto& A = prob.A[n];
            const auto& B = prob.B[n];
            const auto& Q = prob.Q[n];
            const auto& R = prob.R[n];

            // ============ Algorithm 5 ========= //
            if (opt.force_chol_P)
            {   
                // 
                Eigen::MatrixXd U_P = Pn1;
                int infoP = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', nx, U_P.data(), nx);
                if (infoP != 0)
                {
                    throw std::runtime_error("Alg5: dpotrf(P_{n+1}) failed at stage" + std::to_string(n));
                }

                // Re = R + (U_P * B)' * (U_P * B) dtrmm + dsyrk(只写上三角)
                Eigen::MatrixXd Y_B = B;    // in-place Y_B = U_P * B
                cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasNoTrans, CblasNonUnit,
                            nx, nu, 1.0, U_P.data(), U_P.rows(),
                            Y_B.data(), Y_B.rows());
                

                // Re = R + (UnB)^T(UnB)
                Eigen::MatrixXd Re = R;
                cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                  nu, nx, 1.0, Y_B.data(), Y_B.rows(),
                  1.0, Re.data(), Re.rows()); // Re += Y_B'^Y_B

                // chol(Re) = U^T U
                Eigen::MatrixXd U = Re;
                int infoRe = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', nu, U.data(), nu);
                if (infoRe != 0) {
                    if (opt.check_spd) throw std::runtime_error("Alg5: dpotrf(Re) failed at stage " + std::to_string(n));
                    // 可在此加对角正则后重试
                }

                // 4) L = U^{-1} * ( S + (U_P*B)'(U_P*A) )
                Eigen::MatrixXd Y_A = A;
                cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper,
                  CblasNoTrans, CblasNonUnit,
                  nx, nx, 1.0, U_P.data(), U_P.rows(),
                  Y_A.data(), Y_A.rows());   // Y_A = U_P*A
                
                // BtPA = Y_B' * Y_A 放到L里复用内存
                Eigen::MatrixXd L(nu,nx);
                L.setZero();
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                  nu, nx, nx, 1.0,
                  Y_B.data(), Y_B.rows(),
                  Y_A.data(), Y_A.rows(),
                  0.0, L.data(), L.rows());
                if (hasS) L.noalias() += prob.S[n];

                // 解U * L = RHS - L = U^{-1}*RHS
                cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper,
                  CblasNoTrans, CblasNonUnit,
                  nu, nx, 1.0, U.data(), U.rows(), L.data(), L.rows());

                // 5) Pn = Q + (U_P*A)'(U_P*A) - L'^L  —— 两次 dsyrk，只写上三角
                Eigen::MatrixXd Pn = Q;
                cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                  nx, nx, 1.0, Y_A.data(), Y_A.rows(),
                  1.0, Pn.data(), Pn.rows());     // + Y_A'^Y_A
                cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                  nx, nu, -1.0, L.data(), L.rows(),
                  1.0, Pn.data(), Pn.rows());     // - L'^L
                Pn = 0.5*(Pn + Pn.transpose());             // 数值对称化

                
                // 6) 线性项：t, l, p_n
                // t = P_{n+1} b_n + p_{n+1}  —— 只读上三角
                Eigen::VectorXd t(nx);
                if (hasb) {
                    t = pn1;
                    cblas_dsymv(CblasColMajor, CblasUpper,
                    nx, 1.0, Pn1.data(), nx,
                    prob.b[n].data(), 1,
                    1.0, t.data(), 1);
                } else {
                    t = pn1;
                }

                // se = s + B'^t
                Eigen::VectorXd se(nu);
                if (hass) se = prob.s[n];
                else      se.setZero(nu);
                cblas_dgemv(CblasColMajor, CblasTrans,
                  nx, nu, 1.0, B.data(), B.rows(),
                  t.data(), 1, 1.0, se.data(), 1);

                // l = U^{-1} se  （解 U*l=se）
                Eigen::VectorXd l(nu);
                l = se;
                cblas_dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                  nu, U.data(), U.rows(), l.data(), 1);

                // p_n = q + A'^t - L'^l
                Eigen::VectorXd pn = (hasq ? prob.q[n] : Eigen::VectorXd::Zero(nx));
                cblas_dgemv(CblasColMajor, CblasTrans,
                  nx, nx, 1.0, A.data(), A.rows(),
                  t.data(), 1, 1.0, pn.data(), 1);
                Eigen::VectorXd Lt_l(nx); Lt_l.setZero();
                cblas_dgemv(CblasColMajor, CblasTrans,
                  nu, nx, 1.0, L.data(), L.rows(),
                  l.data(), 1, 0.0, Lt_l.data(), 1);
                pn.noalias() -= Lt_l;

                // 7) K = -U^{-T} L,  k = -U^{-T} l  （显式构造，便于与 Schur 接口对齐）
                Eigen::MatrixXd K(nu, nx);
                K = L;
                cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper,
                  CblasTrans, CblasNonUnit,
                  nu, nx, 1.0, U.data(), U.rows(), K.data(), K.rows());
                K *= -1.0;

                Eigen::VectorXd k_vec = l;
                cblas_dtrsv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                  nu, U.data(), U.rows(), k_vec.data(), 1);
                k_vec *= -1.0;

                // 保存 & 准备下一层
                res.P[n] = Pn;      res.p[n] = pn;
                res.K[n] = K;       res.k[n] = k_vec;
                Pn1.swap(Pn);       pn1.swap(pn);

            }
            
            // ======== Algorithm 4 利用P对称性
            if (opt.use_symmetry)
            {
                Eigen::MatrixXd PB(nx, nu);
                // 计算PB，因为P为对称，用dsymm
                symm_left_upper(nx, nu, Pn1.data(), nx, B.data(), B.rows(), PB.data(), PB.rows());

                Eigen::MatrixXd Re = R;
                // Re += B' * PB
                gemmTN(nu, nu, nx, B.data(), B.rows(), PB.data(), PB.rows(), Re.data(), Re.rows(), 1.0, 1.0);
                Re = 0.5 *(Re + Re.transpose());

                Eigen::MatrixXd U = Re;
                int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', nu, U.data(), nu);
                if (info != 0)
                {
                    if (opt.check_spd) throw std::runtime_error("Alg4(no-cholP): dpotrf(Re) failed at stage " + std::to_string(n));
                }
                else
                {
                    // 3) L = U^{-1}*(S + B' (P * A))
                    Eigen::MatrixXd PA(nx, nx);
                    symm_left_upper(nx, nx, Pn1.data(), nx, A.data(), A.rows(), PA.data(), PA.rows());  // PA = P * A

                    Eigen::MatrixXd BtPA(nu, nx);
                    gemmTN(nu, nx, nx, B.data(), B.rows(), PA.data(), PA.rows(), BtPA.data(), BtPA.rows());

                    Eigen::MatrixXd L = hasS ? (BtPA + prob.S[n]) : BtPA;
                    trsm_left_upper('N', nu, nx, U.data(), U.rows(), L.data(), L.rows());   // 解 U * L = RHS

                    // 4) Pn = Q + A' * (P * A) - L' * L
                    Eigen::MatrixXd Pi = P;
                    Pi.triangularView<Eigen::StrictlyLower>().setZero();
                    Pi.diagonal() *= 0.5;

                    // Y_A = Π' * A_n
                    Eigen::MatrixXd Y_A = A;
                    trmm_left_upper('T', nx, nx, Pi.data(), nx, Y_A.data(), Y_A.rows());

                    // T = A' * Y_A
                    Eigen::MatrixXd T(nx, nx);
                    gemmTN(nx, nx, nx, A.data(), A.rows(), Y_A.data(), Y_A.rows(), T.data(), T.rows(), 1.0, 0.0);

                    // Pn = Q + T + T' - L' * L
                    Eigen::MatrixXd Pn = Q;
                    Pn.noalias() += T + T.transpose();
                    Pn.noalias() -= L.transpose() * L;
                    Pn = 0.5 * (Pn + Pn.transpose());

                    // 5) 线性项
                    // t = P * b + p   —— 用 dsymv 只读上三角（若无 b，则 t = p）
                    Eigen::VectorXd t(nx);
                    if (hasb) {
                        t = pn1;
                        symv_upper(nx, Pn1.data(), nx, prob.b[n].data(), 1, t.data(), 1, 1.0, 1.0);
                    } else { t = pn1; }

                    Eigen::VectorXd se(nu);
                    gemvT(nx, nu, B.data(), B.rows(), t.data(), 1, se.data(), 1, 1.0, 0.0);
                    if (hass) se.noalias() += prob.s[n];

                    Eigen::VectorXd l = se;
                    trsv_upper('N', nu, U.data(), U.rows(), l.data(), 1);  // l = U^{-1} se

                    Eigen::VectorXd pn = (hasq ? prob.q[n] : Eigen::VectorXd::Zero(nx));
                    Eigen::VectorXd At_t(nx);
                    gemvT(nx, nx, A.data(), A.rows(), t.data(), 1, At_t.data(), 1, 1.0, 0.0);
                    pn.noalias() += At_t;

                    Eigen::VectorXd Lt_l(nx);
                    gemvT(nu, nx, L.data(), L.rows(), l.data(), 1, Lt_l.data(), 1, 1.0, 0.0);
                    pn.noalias() -= Lt_l;

                    // 6) 显式 K,k（可与 Schur 接口对齐）
                    Eigen::MatrixXd K(nu, nx); K = L;
                    trsm_left_upper('T', nu, nx, U.data(), U.rows(), K.data(), K.rows()); K *= -1.0;
                    Eigen::VectorXd k = l;
                    trsv_upper('T', nu, U.data(), U.rows(), k.data(), 1); k *= -1.0;

                    // 保存并进入下一层
                    res.P[n] = Pn; res.p[n] = pn;
                    res.K[n] = std::move(K); res.k[n] = std::move(k);
                    Pn1.swap(Pn); pn1.swap(pn);
                    continue;  // 本层完成
                }

            }
            
            // =========== Algorithm 3 ========== //
            // 1) Re = R + B' * P_{n+1}' * B, 其中 p_{n+1} = Pn1, PB = P * B
            // 步骤分两步: 先 PB = Pn1 * B(避免重复读写)，再Re = R + B^T * PB
            Eigen::MatrixXd PB(nx ,nu);
            // PB = Pn1 * B
            //   数学含义：PB := P_{n+1} B，维度 (nx x nu)
            //   BLAS 调用：C := alpha*A*B + beta*C
            //     m = nx, n = nu, k = nx
            //     A = Pn1 (nx x nx), lda = nx (= Pn1.rows())
            //     B = B   (nx x nu),  ldb = B.rows() (= nx，Eigen列主序下为行数)
            //     C = PB  (nx x nu),  ldc = PB.rows() (= nx)
            //   复杂度：O(nx * nx * nu)
            gemmNN(nx, nu, nx, Pn1.data(), nx, B.data(), B.rows(), PB.data(), PB.rows());
            Eigen::MatrixXd Re = R;
            // Re = R + Bᵀ * PB
            //   数学含义：Re := R + Bᵀ P_{n+1} B，维度 (nu x nu)
            //   这是等效输入权重 R_e，稍后要对 Re 做 Cholesky 分解（要求 SPD）。
            //   BLAS 调用：C := alpha*Aᵀ*B + beta*C
            //     m = nu, n = nu, k = nx
            //     A = B    (nx x nu),   取 Aᵀ -> (nu x nx), lda = B.rows() (= nx)
            //     B = PB   (nx x nu),                ldb = PB.rows() (= nx)
            //     C = Re   (nu x nu),                ldc = Re.rows() (= nu)
            //     alpha = 1.0, beta = 1.0  （先把 Re 初始化为 R，再累加 Bᵀ*PB）
            //   复杂度：O(nu * nx * nu)
            gemmTN(nu, nu, nx, B.data(), B.rows(), PB.data(), PB.rows(), Re.data(), Re.rows(), 1.0, 1.0);
            

            // 对Re做Cholesky分解，得到上三角因子 U (Re = U^T U)
            // 说明 dpotrf 会原地覆写输入矩阵 
            Eigen::MatrixXd U = Re; // 复制Re, 避免原地覆盖R_e (后面还可能用到Re的完整值)
            // dpotrf: 对称正定矩阵的 Cholesky 分解（double precision）
            // 参数：
            //   layout = LAPACK_COL_MAJOR   // 列主序存储（与 Eigen::MatrixXd 一致）
            //   uplo   = 'U'                // 只使用/填充上三角。最终 U 的上三角是因子，下三角内容未定义。
            //   n      = nu                 // 矩阵阶数（控制维度）
            //   a      = U.data()           // 输入/输出矩阵（原地覆写）；输入是 Re，上三角有效；输出是上三角因子 U
            //   lda    = nu                 // 主维（leading dimension），列主序下等于行数
            // 返回：info = 0 表示成功；>0 表示在该主子式处检测到非正定（不可分解）。
            int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', nu, U.data(), nu);
            if (info != 0)
            {
                // 分解失败常见原因：
                // 1) Re 非对称/数值不够对称（浮点误差），应在分解前对称化：Re = 0.5*(Re + Re.transpose())；
                // 2) Re 不是正定：例如 R 非严格正定、P_{n+1} 半正定导致 Bᵀ P_{n+1} B 仅半正定；
                // 3) 缩放不当/病态（条件数很大），导致数值上判为非正定。
                //
                // 处理建议（按需求选一项）：
                // - 先做数值对称化再分解；
                // - 给对角添加微小扰动（Levenberg-Marquardt 风格）：
                //     Re.diagonal().array() += eps;  // eps 常取 1e-10 ~ 1e-6，视尺度而定
                //   之后重试 dpotrf；
                // - 若算法允许，回退到 Alg3 通用路径或改用更稳健的求解策略；
                // - 检查代价权重：确保 R ≻ 0（严格正定），必要时对 R 加正则；
                // - 检查数值尺度：对 (A,B,Q,R) 做统一缩放，改善条件数。
                if (opt.check_spd) throw std::runtime_error("Alg3 dpotrf(Re) failed at stage " + std::to_string(n));
                    // 若 opt.check_spd == false，可选择忽略并继续（风险：后续三角解将失效），
                    // 更合理的做法是按上面的建议先修复 Re 再继续。  
            }
            // 成功后：U 的上三角部分就是 Cholesky 因子（Re = Uᵀ * U）。
            // 下游应使用“解三角方程”而不是显式求逆：
            //   - 解 U * X = B 或 Uᵀ * X = B 用 dtrsm（矩阵右侧）或 dtrsv（向量右侧）；
            //   - 例如 L = U^{-1} * (S + Bᵀ P A) 可写成：dtrsm(Left, Upper, NoTrans, NonUnit, ...)
            // 复杂度：~ (1/3) * nu^3 flops。对小 nu、长时域 N 的 MPC 来说，这步通常不是瓶颈。

            // L = U^{-1} * (S + (P' B)' A)
            Eigen::MatrixXd PA(nx, nx);
            gemmNN(nx, nx, nx, Pn1.data(), nx, A.data(), A.rows(), PA.data(), PA.rows());
            Eigen::MatrixXd BtPA(nu, nx);
            gemmTN(nu, nx, nx, B.data(), B.rows(), PA.data(), PA.rows(), BtPA.data(), BtPA.rows());
            Eigen::MatrixXd L = hasS ? (BtPA + prob.S[n]) : BtPA;
            // 目的：解线性方程 U * L = M，得到 L = U^{-1} * M。
            // 这里我们把 M 直接存在 L 中（in-place 覆盖），是标准的三角求解写法。
            // 对应数学处：L ← Λ_n^{-1} * (S_n + B_nᵀ P_{n+1} A_n)
            trsm_left_upper('N', nu, nx, U.data(), U.rows(), L.data(), L.rows());

            // Pn = Q + A' P' A - L' L
            Eigen::MatrixXd tmp(nx, nx), LtL(nx, nx);
            gemmTN(nx, nx, nx, A.data(), A.rows(), PA.data(), PA.rows(), tmp.data(), tmp.rows(), 1.0, 0.0);
            gemmTN(nx, nx, nu, L.data(), L.rows(), L.data(), L.rows(), LtL.data(), LtL.rows(), 1.0, 0.0);
            Eigen::MatrixXd Pn = Q + tmp - LtL;
            Pn = 0.5 * (Pn + Pn.transpose());   // P是对称矩阵

            // l, p_n
            // 线性项的中间量: t =  P_{n+1} b_n + p_{n+1}
            Eigen::VectorXd t(nx);
            if (hasb)
            {
                t = pn1;    // 先把p_{n+1}复制到t
                // t ← 1.0 * (P_{n+1} b_n) + 1.0 * t  =>  t = P_{n+1} b_n + p_{n+1}
                // gemvN: y = alpha*A*x + beta*y，其中 A 不转置（NoTrans）
                // 维度：A=P_{n+1} (nx x nx), x=b_n (nx), y=t (nx)
                gemvN(nx, nx, Pn1.data(), nx, prob.b[n].data(), 1, t.data(), 1, 1.0, 1.0);
            }
            else
            {
                // 没有b_n时，t = p_{n+1}
                t = pn1;
            }

            // ========== s 等效项 se = s_n + B_n^T t =====
            Eigen::VectorXd se(nu);
            // se = 1.0 * (B_n^T t) + 0.0
            // gemvT: y = alpha*A^T*x + beta*y（A^T 参与）
            // 维度：A=B_n (nx x nu) → A^T (nu x nx), x=t (nx), y=se (nu)
            gemvT(nx, nu, B.data(), B.rows(), t.data(), 1, se.data(), 1, 1.0, 0.0);
            // 若存在原始s_n: se += s_n
            if (hass) se.noalias += prob.s[n];

            // 求解l l = Λ_n^{-1} * se ====
            // 这里 Λ_n 来自 chol(R_e,n) 的上三角因子 U（Re = U^T U），所以解 U * l = se
            Eigen::VectorXd l = se;
            // trsv_upper('N'): 解 U * l = se（上三角、非单位对角），in-place 覆盖 l
            // 若需要解 U^T * x = y，把 'N' 改为 'T'
            trsv_upper('N', nu, U.data(), U.rows(), l.data(), 1);

            // ===== p_n更新 p_n = q_n + A_n^T t - L_n^T l ======
            Eigen::VectorXd pn = (hasq ? prob.q[n] : Eigen::VectorXd::Zero(nx));

            // pn += A_n^T t
            Eigen::VectorXd At_t(nx);
            // gemvT: At_t = A_n^T *t
            gemvT(nx, nx, A.data(), A.rows(), t.data(), 1, At_t.data(), 1, 1.0, 0.0);
            pn.noalias() += At_t;
            
            // pn -= L_n^T *l
            Eigen::VectorXd Lt_l(nx);
            // L 维度 (nx x nx)，所以L^T (nx x nu)
            // gemvT Lt_l = L^T * l
            gemvT(nu, nx, L.data(), L.rows(), l.data(), 1, Lt_l.data(), 1, 1.0, 0.0);
            pn.noalias() -= Lt_l;

            //   K = -Λ^{-T} L,   k = -Λ^{-T} l
            // 其中 Λ 来自 chol(Re) 的上三角因子：Re = Λᵀ Λ。
            // 推导回顾：u* = K x + k，且 L = Λ^{-1} (S + Bᵀ P A)，l = Λ^{-1} (s + Bᵀ (P b + p))。
            Eigen::MatrixXd k(nu,nx);
            K = L;  // 先把右端设为L，随后in-place解三角系统得到K = Λ^{-T} L
            // // trsm_left_upper('T', ...) 表示 op(Λ) = Λᵀ，左侧上三角三角解，结果覆盖在 K 中
            trsm_left_upper('T', nu, nx, U.data(), U.rows(), K.data(), K.rows());
            // K = -K
            K *= -1.0;

            Eigen::VectorXd k = l;  // 同理，k = -Λ^{-T} l
            // 解 Λᵀ * k = l  →  k = Λ^{-T} l
            trsv_upper('T', nu, U.data(), U.rows(), k.data(), 1);
            k *= -1.0;

            // 现在得到：u_n = K x_n + k，与之前用 U、L、l 的 “-Λ^{-T}(Lx + l)” 完全等价。
            // 说明：显式存 K、k 便于与 Schur 接口对齐；若追求数值/性能，也可在前向时直接用 dtrsv 计算 u，省去存 K、k。





            
            // Re = R
            Re = R;
            // Re += B' * PB
            gemmTN(nu, nu, B.data(), B.rows(), PB.data(), PB.rows(), Re.data(), Re.rows(), 1.0, 1.0);

            // 2) Cholesky: Re = U^T U (上三角)
            Eigen::MatrixXd U = Re; // dpotrf 就地分解
            int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', nu, U.data(), nu);
            if (info != 0)
            {
                if (opt.check_spd)
                {
                    throw std::runtime_error("dpotrf failed at stage " + std::to_string(n) + ", info=" + std::to_string(info));
                }
                // 否则可能退化为加扰动，或其他策略(此处简单抛错/忽略由用户选择)
            }

            // 3) L = U^{-1} * (S + B' * P * A)
            // PA = P * A
            gemmNN(nx, nx, nx, Pn1.data(), nx, A.data(), A.rows(), PA.data(), PA.rows());
            // BtPA = B' * PA
            gemmTN(nu, nx, nx, B.data(), B.rows(), PA.data(), PA.rows(), BtPA.data(), BtPA.rows());
            // Lbuf = S + BtPA
            L = BtPA;
            if (hasS) L.noalias() += prob.S[n];
            // 解 U * L = （S + B' P A) ==> L = U^{-1}
            trsm_left_upper('N', nu, nx, U.data(), U.rows(), L.data(), L.rows());

            // 4) P_n = Q + A' P A - L' L
            // tmp =  A' P A
            gemmTN(nx, nx, nx, A.data(), A.rows(), PA.data(), PA.rows(), tmp.data(), tmp.rows(), 1.0, 1.0);
            Eigen::MatrixXd Pn = Q + tmp;
            // Pn -= L' L
            Eigen::MatrixXd LtL(nx, nx);
            gemmTN(nx, nx, nu, L.data(), L.rows(), L.data(), L.rows(), LtL.data(), LtL.rows(), 1.0, 0.0);
            Pn.noalias() -= LtL;

            // 5) l = U^{-1} * (s + B' * (P b + p))
            if (hasb)
            {
                // t = P b + p
                t = pn1;
                gemvN(nx, nx, Pn1.data(), nx, prob.b[n].data(), 1, t.data(), 1, 1.0, 1.0);

            }
            else
            {
                t = pn1;    // 没有b 即 t = p_{n+1}
            }

            // se = B' t (+ s)
            gemvT(nx, nu, B.data(), B.rows(), t.data(), 1, se.data(), 1, 1.0, 0.0);
            if (hass) se.noalias() += prob.s[n];

            // 解 U * l = se
            Eigen::VectorXd l = se;
            trsv_upper('N', nu, U.data(), U.rows(), l.data(), 1);

            // 6) p_n = q + A' t - L' l
            Eigen::VectorXd pn = (hasq ? prob.q[n] : Eigen::VectorXd::Zero(nx));
            // A' t
            Eigen::VectorXd At_t(nx);
            gemvT(nu, nx, L.data(), L.rows(), l.data(), 1, Lt_l.data(), 1, 1.0, 0.0);
            pn.noalias() -= Lt_l;

            // 保存本层结果
            res.P[n] = Pn;
            res.p[n] = pn;
            // 记录 K，k K=-U^{-T} L, k = -U^{-T} l
            // 实际前向时我们会用U 和L, l直接做三角求解，更稳更快，但也给出了显式K，k
            Eigen::MatrixXd K(nu, nx);
            K = L;  // 先K = L
            // 解 U^{T} * K = L => K = U^{-T} * L
            trsm_left_upper('T', nu, nx, U.data(), U.rows(), K.data(), K.rows());
            K *= -1.0;
            
            Eigen::VectorXd k = l;  // k = l
            trsv_upper('T', nu, U.data(), U.rows(), k.data(), 1);
            k *= -1.0;

            res.K[n] = std::move(K);
            res.k[n] = std::move(k);

            // 更新下一轮
            Pn1.swap(Pn);
            pn1.swap(pn);

            
        
            // 保存递推结果
            res.P[n] = Pn;  
            res.p[n] = pn;
            res.K[n] = std::move(K);
            res.k[n] = std::move(k);
            
            // 为下一层做准备
            // 把当前层得到的(P_n, p_n)作为下一轮的(P_{n+1}, p_{n+1})
            Pn1.swap(Pn);
            pn1.swap(pn);
            // —— 实现与数值注意 ——
            // 1) 使用三角解（trsm/trsv）而非显式求逆更稳健且高效；U 必须来自 Re 的 Cholesky，保证非奇异。
            // 2) 若后续仅需要在线计算 u，可不存 K、k：在前向时直接计算 y = L x + l，再解 Λᵀ u = -y 得到 u（两次 dtrsv）。
            // 3) 维度：U(nu×nu 上三角), L(nu×nx), l(nu), K(nu×nx), k(nu)；nx=状态维，nu=控制维。
            // 4) 若数值误差导致 P 轻微非对称，前面已做 P = 0.5*(P + Pᵀ) 对称化，有利于后续稳定性。
            // 5) move 赋值(std::move)避免一次额外拷贝；swap 将 Pn、pn 作为下一阶段的 P_{n+1}、p_{n+1}，零拷贝切换。
            
        }


        // 终端保存
        res.Pn[N] = prob.P_N;
        res.pn[N] = prob.p_n;

        // 前向rollout
        if (opt.do_forward_pass == true)
        {
            assert_size(prob.x0.size() == nx, "x0 dim mismatch.")
            res.x.resize(N+1);
            res.u.resize(N);
            res.x[0] = prob.x0;

            for (int n=0; n<N; n++)
            {
                const auto& A = prob.A[n];
                const auto& B = prob.B[n];
                const auto& K = prob.K[n];
                const auto& k = prob.k[n];

                // u = -K * x + k
                res.u[n].resize(nu);
                res.u[n].noalias() = K * res.x[n] + k;
                
                // x_{n+1} = A x_{n} + B u_{n} + b
                res.x[n+1].resize(nx);
                res.x[n+1].noalias() = A * res.x[n] + B * res.u[n];
                if (hasb)
                {
                    res.x[n+1].noalias() += prob.b[n];
                }
            }

        }

        return res;

    };



    /*
    矩阵-矩阵乘 gemm
    矩阵-向量乘 gemv

    cblas_dgemm: C = alpha * op(A) * op(B) + beta * C; op()表示原矩阵或者其转置
    m: 结果矩阵C的行数，op(A) 是(m k)
    n: 结果矩阵C的列数 op(B)是(k n)
    k: A的列数(或A^T的行数)，同时也是B的行数，A是(m k), B是(k n)
    lda: A的主维，列主序下等于A的行数 
    ldb: B的主维
    ldc: C的主维

    cblas_dgemv: y = alpha * op(A) * x + beta * y
    m: A的行数 A是(m n)
    n: A的列数 
    x: 输入向量, 若op(A)=A则x长n，若op(A) = A^T,则x长m
    y: 输出向量，若op(A)=A则y长m，若op(A) = A^T,则y长n


    CblasColMajor 列主序存储； CblasRowMajor 行主序存储
    CblasNoTrans 不转置； CblasTrans 转置； CblasConjTrans 共轭转置

    */    
    void RiccatiSolver::gemmNN(int m, int n, int k,
                                const double* A, int lda,
                                const double* B, int ldb,
                                double* Y, int ldy,
                                double alpha, double beta)
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, lda, B, ldb, beta, Y, ldy);
    }

    void RiccatiSolver::gemmTN(int m, int n, int k,
                                const double* A, int lda,
                                const double* B, int ldb,
                                double* Y, int ldy,
                                double alpha, double beta)
    {
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    m, n, k, alpha, A, lda, B, ldb, beta, Y, ldy)
    }

    void RiccatiSolver::gemmNT(int m, int n, int k,
                           const double* A, int lda,
                           const double* B, int ldb,
                           double* Y, int ldy,
                           double alpha, double beta)
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
              m, n, k, alpha, A, lda, B, ldb, beta, Y, ldy);
    }

    void RiccatiSolver::gemvN(int m, int n,
                          const double* A, int lda,
                          const double* x, int incx,
                          double* y, int incy,
                          double alpha, double beta) {
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    void RiccatiSolver::gemvT(int m, int n,
                          const double* A, int lda,
                          const double* x, int incx,
                          double* y, int incy,
                          double alpha, double beta) {
    cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    void RiccatiSolver::trsm_left_upper(char trans, int m, int n,
                                        const double* U, int ldu,
                                        double* B, int ldb)
    {
        CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
        /*
        cblas_dtrsm是用来解三角线性方程组
        U * X = RHS (得到X = U^{-1} RHS) 或
        U^T * X = RHS (得到 X = U^{-T} RHS)
        
        */
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNonUnit,
                m, n, 1.0, U, ldu, B, ldb);
    }

    void RiccatiSolver::trsv_upper(char trans, int n,
                                    const double* U, int ldu,
                                    double* x, int incx)
    {
        CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
        // 解op(U) * x = b 右端是单个向量
        cblas_dtrsv(CblasColMajor, CblasUpper, t, CblasNonUnit, n, U, ldu, x, incx);
    }

    void RiccatiSolver::syrk_upper(char trans, int n, int k,
                                    const double* A, int lda,
                                    double* C, int ldc,
                                    double alpha, double beta)
    {
        // C := alpha * op(A) * op(A)' + beta * C, 仅更新上或下三角，后续需要做数值对称化，比gemm快
        CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
        cblas_dsyrk(CblasColMajor, CblasUpper, t, n, k, alpha, A, lda, beta, C, ldc);
    }

    void RiccatiSolver::trmm_left_upper(char /*unused*/, bool transpose_on_tri,
                                        int m, int n,
                                        const double* U, int ldu,
                                        double* B, int ldb)
    {
        // 计算 B := op(U) * B, 其中 U 是上三角，op(U) = U 或 U^T
        CBLAS_TRANSPOSE t = transpose_on_tri ? CblasTrans : CblasNoTrans;
        // 只做乘法，不解方程
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, t, CblasNonUnit,
                    m, n, 1.0, U, ldu, B, ldb);
    }

    void RiccatiSolver::trmv_upper(char trans, int n,
                                    const double* U, int ldu,
                                    double* x, int incx)
    {
        // x := op(U) * x
        CBLAS_TRANSPOSE t = (trans == 'N' ? CblasNoTrans : CblasTrans);
        // 三角 x 向量 只做乘法，不解方程
        cblas_dtrmv(CblasColMajor, CblasUpper, t, CblasNonUnit, n, U, ldu, x, incx);
    }

    void RiccatiSolver::symv_upper(int n,
                                    const double* UPU, int ldu,
                                    const double* x, int incx,
                                    double* y, int incy,
                                    double alpha, double beta)
    {
        // y := alpha * Sym(UPU) * x + beta * y, 仅使用上三角
        // 对称 x 向量 只做乘法
        cblas_dsymv(CblasColMajor, CblasUpper, n, alpha, UPU, ldu, x, incx, beta, y, incy);
    }
    
    void RiccatiSolver::symm_left_upper(int n, int k,
                                    const double* P, int ldP,
                                    const double* M, int ldM,
                                    double* Y, int ldY,
                                    double alpha, double beta)
    {
        // Y := alpha * Sym(P) * M + beta * Y(只读上三角)
        cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, n, k, alpha, P, ldP, M, ldM, beta, Y, ldY);
    }
                    
}