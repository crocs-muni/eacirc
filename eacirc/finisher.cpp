#include "finisher.h"
#include "CommonFnc.h"
#include <core/logger.h>

void Finisher::ks_test_finish(std::vector<double>& pvalues, unsigned significance_level) {
    if (pvalues.size() < 3) {
        Logger::warning() << "To few samples,  cannot run KS test." << std::endl;
        return;
    }
    Logger::info() << "KS test on p-values, size=" << pvalues.size() << std::endl;

    double KS_critical_alpha_5 = CommonFnc::KSGetCriticalValue(pvalues.size(), significance_level);
    double KS_P_value = CommonFnc::KSUniformityTest(pvalues);
    Logger::info() << "   KS Statistics: " << KS_P_value << std::endl;
    Logger::info() << "   KS critical value " << significance_level << "%: " << KS_critical_alpha_5
                   << std::endl;

    if (KS_P_value > KS_critical_alpha_5) {
        Logger::info() << "   KS is in " << significance_level
                       << "% interval -> uniformity hypothesis rejected." << std::endl;
    } else {
        Logger::info() << "   KS is not in " << significance_level << "% interval -> is uniform."
                       << std::endl;
    }
}
