#ifndef CIRCUIT_GENOME_H
#define CIRCUIT_GENOME_H

//libinclude (galib/GA1DArrayGenome.h)
#include "GA1DArrayGenome.h"
#include "SSGlobals.h"

class CircuitGenome {
public:
  static void Initializer(GAGenome&);
  static int Mutator(GAGenome&, float);
  static float Evaluator(GAGenome&);
  static int Crossover(const GAGenome&, const GAGenome&,GAGenome*, GAGenome*);
public:
	static void ExecuteFromText(string textCircuit, GA1DArrayGenome<unsigned long> *genome);
	static int ParseCircuit(string textCircuit, unsigned long* circuit, int* numLayers, int* intLayerSize, int* outLayerSize);
	static int PrintCircuit(GAGenome &g, string filePath = "", unsigned char usePredictorMask[MAX_OUTPUTS] = NULL, int bPruneCircuit = FALSE);
	static int GetFunctionLabel(unsigned long functionID, unsigned long connections, string* pLabel);
	static int PruneCircuit(GAGenome &g, GAGenome &prunnedG);
	static int GetUsedNodes(GAGenome &g, unsigned char usePredictorMask[MAX_OUTPUTS], unsigned char displayNodes[MAX_GENOME_SIZE]);
	static int HasConnection(unsigned long functionID, unsigned long connectionMask, int fncSlot, int connectionOffset, int bit, int* pbImplicitConnection);
	static int HasImplicitConnection(unsigned long functionID);
	static int IsOperand(unsigned long functionID, unsigned long connectionMask, int fncSlot, int connectionOffset, int bit, string* pOperand);
	static int GetNeutralValue(unsigned long functionID, string* pOperand);
	static int ExecuteCircuit(GA1DArrayGenome<unsigned long>* pGenome, unsigned char inputs[MAX_INPUTS], unsigned char outputs[MAX_OUTPUTS]);  
};
#endif
