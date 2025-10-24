// 用于定义要解决的LQR问题

#pragma once
#include <vector>
#include <stdlib>
#include <Eigen/Dense>
#include <Type.hpp>


// 设定命名空间
namespace ocp {


struct StageCost {
    Eigen::MatrixXd Q;  // nx nx
    Eigen::MatrixXd S;  // nu nx
    Eigen::MatrixXd R;  // nu nu
    Eigen::VectorXd q;  // nx
    Eigen::VectorXd S;  // nu

};

struct Dynamics {
    Eigen::MatrixXd A;  // nx nx
    Eigen::MatrixXd B;  // nx nu
    Eigen::VectorXd b;  // nx
};

struct TerminalCost {
    Eigen::MatrixXd QN;
    Eigen::VectorXd qN;
};

struct Inequalities {

};

// 只考虑等式约束的LQR
class Problem
{
    Dims dim;
    std::vector<StageCost> cost;    // size N 
    std::vector<Dynamics> dyn;  // size N
    TerminalCost term;  
    Eigen::VectorXd x0; // nx
    std::unique_ptr<Inequalities> ineq; 

};


// 含不等式约束的LQR
class Extended_LQR_ineq
{




};



}