#pragma once

#include <sstream>
#include <stdexcept>

namespace core {

struct assertion_failure : std::logic_error {
    assertion_failure(const char *function, const char *file, int line)
        : std::logic_error(static_cast<std::ostringstream &>(
                                   std::ostringstream{}.flush()
                                   << "Assertion failure in function "
                                   << function << "in " << file << ":" << line)
                                   .str()) {}
};

#define ASSERT_ALLWAYS(expr_)                                                  \
    if (!(expr_))                                                              \
        throw ::core::assertion_failure{__FUNCTION__, __FILE__, __LINE__};

#ifdef NDEBUG
#define ASSERT(expr_)
#else
#define ASSERT(expr_) ASSERT_ALLWAYS(expr_)
#endif

#define ASSERT_UNREACHABLE() ASSERT(false)

} // namespace core
