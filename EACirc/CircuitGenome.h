#ifndef CIRCUIT_GENOME_H
#define CIRCUIT_GENOME_H

//libinclude (galib/GA1DArrayGenome.h)
#include "GA1DArrayGenome.h"
#include "EACglobals.h"

class CircuitGenome {
public:
    static void Initializer(GAGenome&);
    static int Mutator(GAGenome&, float);
    static float Evaluator(GAGenome&);
    static int Crossover(const GAGenome&, const GAGenome&,GAGenome*, GAGenome*);
    static void ExecuteFromText(string textCircuit, GA1DArrayGenome<unsigned long>* genome);
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

    /** saves genome to string in binary format
      * @param genome       genome to print
      * @param textCircuit  printed genome (original contents overwritten)
      * @return status
      */
    static int writeGenome(const GA1DArrayGenome<unsigned long>& genome, string& textCircuit);

    /** reads genome in binary fromat from string
      * genome parameters (number of layers, genome size, etc.) are taken from main settings
      * - genome is prolonged by zeroes if needed
      * @param genome       read genome (contents overwritten)
      * @param textCircuit  string to read circuit from
      * @return status
      */
    static int readGenome(GA1DArrayGenome<unsigned long>& genome, string& textCircuit);

private:
    /** saves circuit as generation with size 1
      * - CAREFUL: code duplicity with Eacirc.cpp!
      * @param genome       genome to save
      * @param filemane     destination filename
      * @return status
      */
    static int saveCircuitAsPopulation(const GA1DArrayGenome<unsigned long> &genome, const string filename);
};
#endif
