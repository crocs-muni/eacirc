#pragma once

#include "dataset.h"

struct Stream {
    virtual ~Stream() = default;
    virtual void read(Dataset&) = 0;
};
