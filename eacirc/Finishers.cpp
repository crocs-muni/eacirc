#include "Finishers.h"
#include "CommonFnc.h"
#include "circuit/ICircuit.h"

void Finishers::avgFitnessFinisher() {
    mainLogger.out(LOGGER_INFO) << "Cumulative results for this run:" << endl << setprecision(FITNESS_PRECISION_LOG);
    mainLogger.out(LOGGER_INFO) << "   AvgAvg: " << pGlobals->stats.avgAvgFitSum / (double) pGlobals->stats.avgCount << endl;
    mainLogger.out(LOGGER_INFO) << "   AvgMax: " << pGlobals->stats.avgMaxFitSum / (double) pGlobals->stats.avgCount << endl;
    mainLogger.out(LOGGER_INFO) << "   AvgMin: " << pGlobals->stats.avgMinFitSum / (double) pGlobals->stats.avgCount << endl;
}

void Finishers::ksUniformityTestFinisher() {
    const unsigned long pvalsSize = pGlobals->stats.pvaluesBestIndividual->size();
    if (pvalsSize <= 2) {
        mainLogger.out(LOGGER_WARNING) << "Only 1-2 samples, cannot run K-S test." << endl;
        return;
    }
    mainLogger.out(LOGGER_INFO) << "KS test on p-values, size=" << pvalsSize << endl;

    double KS_critical_alpha_5 = CommonFnc::KSGetCriticalValue(pvalsSize, pGlobals->settings->main.significanceLevel);
    double KS_P_value = CommonFnc::KSUniformityTest(*(pGlobals->stats.pvaluesBestIndividual));
    mainLogger.out(LOGGER_INFO) << "   KS Statistics: " << KS_P_value << endl;
    mainLogger.out(LOGGER_INFO) << "   KS critical value " << pGlobals->settings->main.significanceLevel << "%: " << KS_critical_alpha_5 << endl;

    if(KS_P_value > KS_critical_alpha_5) {
        mainLogger.out(LOGGER_INFO) << "   KS is in " << pGlobals->settings->main.significanceLevel << "% interval -> uniformity hypothesis rejected." << endl;
    } else {
        mainLogger.out(LOGGER_INFO) << "   KS is not in " << pGlobals->settings->main.significanceLevel << "% interval -> is uniform." << endl;
    }
}

void Finishers::outputCircuitFinisher(GAGenome &genome) {
    pGlobals->circuit->io()->outputGenomeFiles(genome, FILE_CIRCUIT_DEFAULT);
    GAGenome genomeProccessed = genome;
    if (pGlobals->circuit->postProcess(genome, genomeProccessed)) {
        pGlobals->circuit->io()->outputGenomeFiles(genomeProccessed, string(FILE_CIRCUIT_DEFAULT) + FILE_POSTPROCCESSED_SUFFIX);
    }
}
