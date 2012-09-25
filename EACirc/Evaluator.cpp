#include "CircuitGenome.h"
#include "Evaluator.h"
#include "SSGlobals.h"
#include "test_vector_generator/ITestVectGener.h"
#include "EACirc.h"
#include <string>

Evaluator::Evaluator() {
	this->TVG = new ITestVectGener();
	this->TVG = (ITestVectGener*)TVG->getGenerClass();
	generateTestVectors();
}

void Evaluator::generateTestVectors() {
	TVG->generateTestVectors();
}

int Evaluator::evaluateStep(GA1DArrayGenome<unsigned long> genome, int actGener) {

	pGACirc->bestGenerFit = 0;

	float bestFit = CircuitGenome::Evaluator(genome);
	//float bestFit = genome.score();

	ofstream bestfitfile("bestfitgraph.txt", ios::app);
	ofstream avgfitfile("avgfitgraph.txt", ios::app);
	bestfitfile << actGener << "," << bestFit << endl;
	if (pGACirc->numAvgGenerFit > 0) avgfitfile << actGener << "," << pGACirc->avgGenerFit / pGACirc->numAvgGenerFit << endl;
	else avgfitfile << actGener << "," << "division by zero!!" << endl;
	bestfitfile.close();
	avgfitfile.close();

	ostringstream os2;
    os2 << "(" << actGener << " gen.): " << pGACirc->avgGenerFit << "/" << pGACirc->numAvgGenerFit << " avg, " << bestFit << " best, avgPredict: " << pGACirc->avgPredictions / pGACirc->numAvgGenerFit << ", totalBest: " << pGACirc->maxFit << endl;
    string message = os2.str();
	// SAVE FITNESS PROGRESS
	ofstream out("EAC_fitnessProgress.txt", ios::app);
	out << message << endl;
	out.close();

	pGACirc->avgGenerFit = 0;
	pGACirc->numAvgGenerFit = 0;
	pGACirc->avgPredictions = 0;

	return 0;
}
