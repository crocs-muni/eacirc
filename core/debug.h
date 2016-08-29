#pragma once

#include <sstream>
#include <stdexcept>

namespace debug {

    struct assertion_failure : std::logic_error {
        assertion_failure(const char* message)
            : std::logic_error(message) {}
    };

} // namespace debug

#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)

#define ASSERT_ALLWAYS(expr_)                                                                      \
    if (!(expr_))                                                                                  \
        throw ::debug::assertion_failure("Assertion failure in function " STRINGIFY(               \
                __FUNCTION__) " in " __FILE__ ":" STRINGIFY(__LINE__));

#ifdef NDEBUG
#define ASSERT(expr_)
#else
#define ASSERT(expr_) ASSERT_ALLWAYS(expr_)
#endif

#define ASSERT_UNREACHABLE() ASSERT(false)
