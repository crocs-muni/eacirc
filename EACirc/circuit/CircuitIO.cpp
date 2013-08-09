#include "CircuitIO.h"
#include "XMLProcessor.h"
#include "CommonFnc.h"

int CircuitIO::genomeFromBinary(string binaryCircuit, GA1DArrayGenome<GENOME_ITEM_TYPE>& genome) {
    istringstream circuitStream(binaryCircuit);
    GENOME_ITEM_TYPE gene;
    for (int offset = 0; offset < pGlobals->settings->circuit.genomeSize; offset++) {
        circuitStream >> gene;
        if (circuitStream.fail()) {
            mainLogger.out(LOGGER_ERROR) << "Cannot load binary genome - error at offset " << offset << "." << endl;
            return STAT_DATA_CORRUPTED;
        }
        genome.gene(offset, gene);
    }
    return STAT_OK;
}

int CircuitIO::genomeFromText(string filename, GA1DArrayGenome<GENOME_ITEM_TYPE>& genome) {

}

int CircuitIO::outputGenomeFiles(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, string fileName) {
    int status = STAT_OK;
    status = genomeToText(genome, fileName + ".txt");
    if (status != STAT_OK) mainLogger.out(LOGGER_WARNING) << "Problem writing text genome (" << statusToString(status) << ")." << endl;
    status = genomeToPopulation(genome, fileName + ".xml");
    if (status != STAT_OK) mainLogger.out(LOGGER_WARNING) << "Problem writing binary genome (" << statusToString(status) << ")." << endl;
    status = genomeToGraph(genome, fileName + ".dot");
    if (status != STAT_OK) mainLogger.out(LOGGER_WARNING) << "Problem writing graph genome (" << statusToString(status) << ")." << endl;
    status = genomeToCode(genome, fileName + ".c");
    if (status != STAT_OK) mainLogger.out(LOGGER_WARNING) << "Problem writing code genome (" << statusToString(status) << ")." << endl;
    return STAT_OK;
}

int CircuitIO::genomeToBinary(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, string& binaryCircuit) {
    int status = STAT_OK;
    ostringstream textCicruitStream;
    for (int i = 0; i < genome.length(); i++) {
        textCicruitStream << genome.gene(i) << " ";
        if (i % pGlobals->settings->circuit.sizeLayer == pGlobals->settings->circuit.sizeLayer - 1) {
            textCicruitStream << "  ";
        }
    }
    binaryCircuit = textCicruitStream.str();
    return status;
}

int CircuitIO::genomeToPopulation(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, string fileName) {
    int status = STAT_OK;
    TiXmlElement* pRoot = populationHeader(1);
    TiXmlElement* pElem = NULL;
    TiXmlElement* pElem2 = NULL;

    pElem = new TiXmlElement("population");
    string textCircuit;
    status = genomeToBinary(genome, textCircuit);
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

int CircuitIO::genomeToText(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, string fileName) {

}

int CircuitIO::genomeToCode(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, string fileName) {

}

int CircuitIO::genomeToGraph(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, string fileName) {

}

TiXmlElement* CircuitIO::populationHeader(int populationSize) {
    TiXmlElement* pRoot = new TiXmlElement("eacirc_population");
    TiXmlElement* pElem = NULL;
    TiXmlElement* pElem2 = NULL;

    pElem = new TiXmlElement("population_size");
    pElem->LinkEndChild(new TiXmlText(toString(populationSize).c_str()));
    pRoot->LinkEndChild(pElem);
    pElem = new TiXmlElement("circuit_dimensions");
    pElem2 = new TiXmlElement("num_layers");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->circuit.numLayers).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("size_layer");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->circuit.sizeLayer).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("size_input_layer");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->circuit.sizeInputLayer).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("size_output_layer");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->circuit.sizeOutputLayer).c_str()));
    pElem->LinkEndChild(pElem2);
    pRoot->LinkEndChild(pElem);

    return pRoot;
}

GENOME_ITEM_TYPE CircuitIO::relativeToAbsoluteConnectorMask(GENOME_ITEM_TYPE relativeMask, int slot, int numLayerConnectors) {

}

GENOME_ITEM_TYPE CircuitIO::absoluteToRelativeConnectorMask(GENOME_ITEM_TYPE absoluteMask, int slot, int numLayerConnectors) {

}
