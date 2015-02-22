#include "PolynomialCircuitIO.h"
#include "XMLProcessor.h"
#include "CommonFnc.h"
#include "circuit/ICircuitIO.h"
#include "Term.h"
#include "PolynomialCircuit.h"

int PolyIO::genomeToBinary_static(GAGenome& g, string& binaryCircuit) {
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>& genome = dynamic_cast<GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>&>(g);
    int numVariables = PolynomialCircuit::getNumVariables();
    int numPolynomials = PolynomialCircuit::getNumPolynomials();
    unsigned int   termSize = Term::getTermSize(numVariables);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.

    int status = STAT_OK;
    ostringstream textCicruitStream;
    for (int i = 0; i < numPolynomials; i++) {
        // Get number of terms in the genome.
        POLY_GENOME_ITEM_TYPE numTerms = genome.gene(i, 0);
        textCicruitStream << setw(3) << right << setfill('0') << numTerms << " ";

        for(unsigned int j = 0; j < (numTerms * termSize); j++){
            textCicruitStream << setw(3) << right << setfill('0') << genome.gene(i, 1+j) << " ";
        }

        textCicruitStream << "      ";
    }
    binaryCircuit = textCicruitStream.str();
    return status;
}

int PolyIO::genomeFromBinary_static(string binaryCircuit, GAGenome& g) {
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>& genome = dynamic_cast<GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>&>(g);
    int numVariables = PolynomialCircuit::getNumVariables();
    int numPolynomials = PolynomialCircuit::getNumPolynomials();
    unsigned int   termSize = Term::getTermSize(numVariables);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.

    istringstream circuitStream(binaryCircuit);
    POLY_GENOME_ITEM_TYPE gene;

    unsigned long ctr = 0;
    int cPoly = 0;
    for(; cPoly < numPolynomials; cPoly++){
        // Get length of the given polynomial
        POLY_GENOME_ITEM_TYPE numTerms;

        circuitStream >> numTerms;
        if (circuitStream.fail()) {
            mainLogger.out(LOGGER_ERROR) << "Cannot load binary genome - error at offset " << ctr << "." << endl;
            return STAT_DATA_CORRUPTED;
        }

        for (unsigned int offset = 0; offset < (numTerms * termSize); offset++) {
            circuitStream >> gene;
            if (circuitStream.fail()) {
                mainLogger.out(LOGGER_ERROR) << "Cannot load binary genome - error at offset " << offset << "." << endl;
                return STAT_DATA_CORRUPTED;
            }

            genome.gene(cPoly, 1+offset, gene);
        }
    }
    return STAT_OK;
}

int PolyIO::genomeFromText_static(string filename, GAGenome& g) {
    return STAT_NOT_IMPLEMENTED_YET;
}

int PolyIO::outputGenomeFiles_static(GAGenome& g, string fileName) {
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>& genome = dynamic_cast<GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>&>(g);

    int status = STAT_OK;
    status = genomeToText_static(genome, fileName + ".txt");
    if (status != STAT_OK) mainLogger.out(LOGGER_WARNING) << "Problem writing text genome (" << statusToString(status) << ")." << endl;
    status = genomeToPopulation_static(genome, fileName + ".xml");
    if (status != STAT_OK) mainLogger.out(LOGGER_WARNING) << "Problem writing binary genome (" << statusToString(status) << ")." << endl;
    status = genomeToGraph_static(genome, fileName + ".dot");
    if (status != STAT_OK) mainLogger.out(LOGGER_WARNING) << "Problem writing graph genome (" << statusToString(status) << ")." << endl;
    // TODO/TBD: implement code genomes and uncomment
    //status = genomeToCode(genome, fileName + ".c");
    //if (status != STAT_OK) mainLogger.out(LOGGER_WARNING) << "Problem writing code genome (" << statusToString(status) << ")." << endl;
    return STAT_OK;
}

int PolyIO::genomeToPopulation_static(GAGenome& g, string fileName) {
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>& genome = dynamic_cast<GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>&>(g);

    int status = STAT_OK;
    TiXmlElement* pRoot = populationHeader_static(1);
    TiXmlElement* pElem = NULL;
    TiXmlElement* pElem2 = NULL;

    pElem = new TiXmlElement("population");
    string textCircuit;
    status = genomeToBinary_static(genome, textCircuit);
    if (status != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "Could not save circuit to file " << fileName << "." << endl;
        return status;
    }
    pElem2 = new TiXmlElement("genome");
    pElem2->LinkEndChild(new TiXmlText(textCircuit.c_str()));
    pElem->LinkEndChild(pElem2);
    pRoot->LinkEndChild(pElem);

    status = saveXMLFile(pRoot, fileName);
    if (status != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "Cannot save circuit to file " << fileName << "." << endl;
        return status;
    }
    return status;
}

int PolyIO::genomeToText_static(GAGenome& g, string fileName) {
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>& genome = dynamic_cast<GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>&>(g);

    int numVariables = PolynomialCircuit::getNumVariables();
    int numPolynomials = PolynomialCircuit::getNumPolynomials();
    unsigned int   termSize = Term::getTermSize(numVariables);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.

    ofstream file(fileName);
    if (!file.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot write genome (" << fileName << ")." << endl;
        return STAT_FILE_WRITE_FAIL;
    }

    // output file header with current circuit configuration
    file << pGlobals->settings->main.circuitSizeInput << " \t(number of variables)" << endl;
    file << pGlobals->settings->main.circuitSizeOutput << " \t(number of polynomials)" << endl;
    file << pGlobals->settings->polyCircuit.numPolynomials << " \t(number of polynomials)" << endl;
    file << pGlobals->settings->polyCircuit.mutateTermStrategy << " \t(term mutation strategy)" << endl;
    file << pGlobals->settings->polyCircuit.maxNumTerms << " \t(maximum number of terms in polynomial)" << endl;
    file << pGlobals->settings->polyCircuit.genomeInitTermCountProbability << " \t(p for geometric distribution for number of variables in term)" << endl;
    file << pGlobals->settings->polyCircuit.genomeInitTermStopProbability << " \t(p for geometric distribution for number of terms in polynomial)" << endl;
    file << pGlobals->settings->polyCircuit.mutateAddTermProbability << "\t(p for adding a new term in a mutation)" << endl;
    file << pGlobals->settings->polyCircuit.mutateAddTermStrategy << "\t(strategy for adding a new term in a mutation)" << endl;
    file << pGlobals->settings->polyCircuit.mutateRemoveTermProbability << "\t(p for removing a random term in a mutation)" << endl;
    file << pGlobals->settings->polyCircuit.mutateRemoveTermStrategy << "\t(strategy for removing a random term in a mutation)" << endl;
    file << pGlobals->settings->polyCircuit.crossoverRandomizePolySelect << "\t(use random permutation in polynomials in crossover)" << endl;
    file << pGlobals->settings->polyCircuit.crossoverTermsProbability << "\t(p for crossing individual terms using single point crossover)" << endl;
    file << endl;

    // output circuit itself, starting with input pseudo-layer
    /*for (int slot = 0; slot < numPolynomials; slot++) {
        file << "IN  [" << setw(2) << right << setfill('0') << slot << "]" << "   ";
    }
    file << endl << endl;*/

    int status = STAT_OK;
    ostringstream textCicruitStream;
    for (int cPoly = 0; cPoly < numPolynomials; cPoly++) {
        file << "#" << setw(2) << right << setfill('0') << (int) cPoly << ": ";

        // Get number of terms in the genome.
        POLY_GENOME_ITEM_TYPE numTerms = genome.gene(cPoly, 0);
        file << " [#" << setw(2) << right << setfill('0') << (int) numTerms << "]" << "   ";

        // Read term by term
        for(unsigned int cTerm = 0; cTerm < numTerms; cTerm++){

            // Read sub-terms
            for(unsigned int j = 0; j < termSize; j++){
                POLY_GENOME_ITEM_TYPE gene = genome.gene(cPoly, 1 + cTerm * termSize + j);

                // Print x_i
                for(unsigned int x = 0; x < 8*sizeof(POLY_GENOME_ITEM_TYPE); x++){
                    if ((gene & (1ul << x)) == 0) continue;

                    file << "x_" << setw(2) << right << setfill('0') << (j*8*sizeof(POLY_GENOME_ITEM_TYPE) + x) << ".";
                }
            }

            // 1 is in each term.
            file << "001";

            // Last term?
            if (cTerm+1 == numTerms){
                file << endl;
            } else {
                file << " + ";
            }
        }
    }

    file << endl;
    file.close();
    return STAT_OK;
}

int PolyIO::genomeToCode_static(GAGenome& g, string fileName) {
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>& genome = dynamic_cast<GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>&>(g);

    return STAT_NOT_IMPLEMENTED_YET;
}

int PolyIO::genomeToGraph_static(GAGenome& g, string fileName) {
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>& genome = dynamic_cast<GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>&>(g);

    return STAT_OK;
}

TiXmlElement* PolyIO::populationHeader_static(int populationSize) {
    TiXmlElement* pRoot = new TiXmlElement("eacirc_population");
    TiXmlElement* pElem = NULL;
    TiXmlElement* pElem2 = NULL;

    pElem = new TiXmlElement("population_size");
    pElem->LinkEndChild(new TiXmlText(toString(populationSize).c_str()));
    pRoot->LinkEndChild(pElem);
    pElem = new TiXmlElement("circuit_dimensions");
    pElem2 = new TiXmlElement("size_input_layer");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->main.circuitSizeInput).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("size_output_layer");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->main.circuitSizeOutput).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("size_memory");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->gateCircuit.sizeMemory).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("polydist_num_polynomials");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->polyCircuit.numPolynomials).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("polydist_max_terms");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->polyCircuit.maxNumTerms).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("polydist_term_count_p");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->polyCircuit.genomeInitTermCountProbability).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("polydist_term_size_p");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->polyCircuit.genomeInitTermStopProbability).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("polydist_mutate_term_strategy");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->polyCircuit.mutateTermStrategy).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("polydist_mutate_add_term_p");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->polyCircuit.mutateAddTermProbability).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("polydist_mutate_add_term_strategy");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->polyCircuit.mutateAddTermStrategy).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("polydist_mutate_remove_term_p");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->polyCircuit.mutateRemoveTermProbability).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("polydist_mutate_remove_term_strategy");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->polyCircuit.mutateRemoveTermStrategy).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("polydist_crossover_randomize_poly");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->polyCircuit.crossoverRandomizePolySelect).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("polydist_crossover_terms_p");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->polyCircuit.crossoverTermsProbability).c_str()));
    pElem->LinkEndChild(pElem2);
    pRoot->LinkEndChild(pElem);
    return pRoot;
}


int PolyIO::genomeFromBinary(string binaryCircuit, GAGenome& g){
    return genomeFromBinary_static(binaryCircuit, g);
}

int PolyIO::genomeFromText(string filename, GAGenome& g){
    return genomeFromText_static(filename, g);
}

int PolyIO::genomeToBinary(GAGenome& g, string& binaryCircuit){
    return genomeToBinary_static(g, binaryCircuit);
}

int PolyIO::genomeToCode(GAGenome& g, string fileName){
    return genomeToCode_static(g, fileName);
}

int PolyIO::genomeToGraph(GAGenome& g, string fileName){
    return genomeToGraph_static(g, fileName);
}

int PolyIO::genomeToPopulation(GAGenome& g, string fileName){
    return genomeToPopulation_static(g, fileName);
}

int PolyIO::genomeToText(GAGenome& g, string fileName) {
    return genomeToText_static(g, fileName);
}

int PolyIO::outputGenomeFiles(GAGenome& g, string fileName){
    return outputGenomeFiles_static(g, fileName);
}

TiXmlElement* PolyIO::populationHeader(int populationSize){
    return populationHeader_static(populationSize);
}
