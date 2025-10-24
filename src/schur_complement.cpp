#pragma once

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <cblas.h>
#include <lapacke.h>
#include "schur_complement.hpp"

namespace OCP
{

// 输入检查
void SchurBackend::validate(const Problem& P)
{

}

// 核心求解
/*
求解KKT方程组
    A * H^-1 * A (phi) = A * H^-1 * g + b
    H * x = A^T * (phi) - g
*/
Status SchurBackend::solve_unconstrained(const Problem& P, Solution& sol)
{
    try{
        validate(P);
        const int N = P.dims.N;
        const int nx = P.dims.nx;
        const int nu = P.dims.nu;
        
        // 计算U矩阵
        // 先Cholysky分解H，分阶段
        // R0 = U0_22^T * U0_22   （直接拿上三角 U0_22）
        Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> lltR0(P.cost[0].R)
        if (lltR0.info()!=Eigen::Success)
        {
            return {Status::kNUMERICS, "R0 not SPD"};
        }
        Eigen::MatrixXd U0_22 = lltR0.matrixU() // 上三角

        // 阶段H = [Q, S^T; S, R]
        std::vector<StageU> UF(N);
        for (int i=0; i<N; i++)
        {
            Eigen::MatrixXd H(nx+nu, nx+nu);
            H.setZero();
            H.topLeftCorner(nx, nu) = p.cost[n].Q;
            H.topRightCorner(nx, nu) = p.cost[n].S.transpose();
            H.bottomLeftCorner(nu,nx) = p.cost[n].S;
            H.bottomRightCorner(nu,nu) = p.cost[n].R;

            Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> llt(H);
            if (llt.info()!=Eigen::Success) return {Status::kNUMERICS, "stage H not SPD"};

            const Eigen::MatrixXd U = llt.matrixU();         // —— 上三角 U
            UF[n].U11 = U.topLeftCorner(nx,nx);
            UF[n].U12 = U.topRightCorner(nx,nu);
            UF[n].U22 = U.bottomRightCorner(nu,nu);

        }

        // 终端阶段 QN = U_N,11^T U_N,11
        Eigen::LLT<Eigen::MatrixXd> lltQN(p.term.QN);
        if (lltQN.info()!=Eigen::Success)
            {
                return Status::kNUMERICS, "QN not SPD"
            }
        Eigen::MatrixXd UN_11 = lltQN.matrixU()



        // 计算U的逆矩阵
        


        // 计算phi

        // 计算psi

        // 对psi进行分解

        // 计算beta

        // 计算pi

        // 计算u和x
        

    }
}





















}