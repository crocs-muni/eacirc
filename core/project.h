#pragma once

#include "testvectors.h"

struct Project {
    virtual void generate(TestVectors&) = 0;
};
