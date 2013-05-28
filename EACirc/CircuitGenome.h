#ifndef CIRCUIT_GENOME_H
#define CIRCUIT_GENOME_H

//libinclude (galib/GA1DArrayGenome.h)
#include "GA1DArrayGenome.h"
#include "EACglobals.h"
#include "tinyxml.h"

/*
	INPUTS
	IN_SELECTOR_LAYER (full interconnection, up to circuit)	
	FUNCTION_LAYER_1 (functions)			

	CONNECTOR_LAYER_2 (number of connectors limited to SETTINGS_CIRCUIT::numConnectors)
	FUNCTION_LAYER_2			
	...
	CONNECTOR_LAYER_N  (number of connectors limited to SETTINGS_CIRCUIT::numConnectors)
	FUNCTION_LAYER_N			

	OUT_SELECTOR_LAYER (full interconnection)	
	OUTPUTS
*/

class CircuitGenome {
public:
    static void Initializer(GAGenome&);

	/** Initialize genom (circuit) into following structire:
		1. IN_SELECTOR_LAYER connects inputs to corresponding FNC in the same column (FNC_1_3->IN_3) 
		2. FUNCTION_LAYER_1 is set to XOR instruction only
		3. CONNECTOR_LAYER_i is set to random mask (possibly multiple connectors)
		4. FUNCTION_LAYER_i is set to random instruction from range 0..FNC_MAX, additionally respecting set of allowed instructions in SETTINGS_CIRCUIT::allowedFunctions 
		   - function argument1 is set to 0
	*/
    static void Initializer_basic(GAGenome&);

	static int Mutator(GAGenome&, float);
    static float Evaluator(GAGenome&);
    static int Crossover(const GAGenome&, const GAGenome&,GAGenome*, GAGenome*);
    static int Crossover_perLayer(const GAGenome&, const GAGenome&,GAGenome*, GAGenome*);
    static int Crossover_perColumn(const GAGenome&, const GAGenome&,GAGenome*, GAGenome*);

    /** reads genome in binary fromat from string
      * genome parameters (number of layers, genome size, etc.) are taken from main settings
      * @param textCircuit  string to read circuit from
      * @param genome       read genome (contents overwritten)
      * @return status
      */
    static int readGenomeFromBinary(string textCircuit, GA1DArrayGenome<GENOM_ITEM_TYPE>* genome);
    static int readGenomeFromText(string textCircuit, GA1DArrayGenome<GENOM_ITEM_TYPE>* genome);

    static int PrintCircuit(GAGenome &g, string filePath = "", unsigned char* usePredictorMask = NULL, int bPruneCircuit = FALSE);
    static int PrintCircuitMemory(GAGenome &g, string filePath = "", unsigned char* usePredictorMask = NULL, int bPruneCircuit = FALSE);
	static int GetFunctionLabel(GENOM_ITEM_TYPE functionID, GENOM_ITEM_TYPE connections, string* pLabel);
	static int PruneCircuit(GAGenome &g, GAGenome &prunnedG);
    static int GetUsedNodes(GAGenome &g, unsigned char* usePredictorMask, unsigned char displayNodes[]);
	static int HasConnection(GENOM_ITEM_TYPE functionID, GENOM_ITEM_TYPE connectionMask, int fncSlot, int connectionOffset, int bit);
	//static int HasImplicitConnection(GENOM_ITEM_TYPE functionID);
	static int IsOperand(GENOM_ITEM_TYPE functionID, GENOM_ITEM_TYPE connectionMask, int fncSlot, int connectionOffset, int bit, string* pOperand);
	static int GetNeutralValue(GENOM_ITEM_TYPE functionID, string* pOperand);
    static int ExecuteCircuit(GA1DArrayGenome<GENOM_ITEM_TYPE>* pGenome, unsigned char* inputs, unsigned char* outputs);

    /** saves genome to string in binary format
      * @param genome       genome to print
      * @param textCircuit  printed genome (original contents overwritten)
      * @return status
      */
    static int writeGenome(const GA1DArrayGenome<GENOM_ITEM_TYPE>& genome, string& textCircuit);

    /** allocate XML structure for header in population file
      * @param populationSize       size of the population (info in the header)
      * @return pointer to root element "eacirc_population"
      */
    static TiXmlElement* populationHeader(int populationSize);

private:
    /** saves circuit as generation with size 1
      * - CAREFUL: code duplicity with Eacirc.cpp!
      * @param genome       genome to save
      * @param filemane     destination filename
      * @return status
      */
    static int saveCircuitAsPopulation(const GA1DArrayGenome<GENOM_ITEM_TYPE> &genome, const string filename);
	
	static unsigned char GET_FNC_TYPE(GENOM_ITEM_TYPE fncValue) {
		return fncValue & 0xff;
	}
	static unsigned char GET_FNC_ARGUMENT1(GENOM_ITEM_TYPE fncValue) {
		return ((fncValue & 0xff000000) >> 24)  & 0xff;
	}
	static void SET_FNC_TYPE(GENOM_ITEM_TYPE* fncValue, unsigned char fncType) {
		*fncValue = *fncValue & 0xffffffff00;
		*fncValue |= fncType;
	}
	static void SET_FNC_ARGUMENT1(GENOM_ITEM_TYPE* fncValue, unsigned char arg1) {
		*fncValue = *fncValue & 0x00ffffffff;
		*fncValue |= arg1 << 24;
	}

};
#endif
