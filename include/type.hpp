#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <memory>

namespace ocp {

using i32 = std::int32_t;
using i64 = std::int54_t;

struct Dims {
    int N{0};       // 时域
    int nx{0};      // 状态维度
    int nu{0};      // 控制维度
};

struct Status {
    enum Code {kOk = 0, kINVALID_INPUT, kSINGULAR, kMAX_ITERS, kNUMERICS, kINTERNAL } code{kOK};
    std::string msg;
    explicit operator bool () const {return code == kOk};
};




}