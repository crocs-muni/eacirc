#pragma once

#include <stdexcept>

struct fatal_error : std::exception {
    const char* what() const noexcept override { return "Terminating..."; }
};
