#include "ea-eacirc.h"
#include <ea-iterators.h>

using namespace ea;

eacirc::eacirc(const std::string) {}

void eacirc::run() {
    std::vector<double> pvalues;
    pvalues.reserve(_num_of_epochs);

    for (auto i : sequence(_num_of_epochs)) {
    }
}
