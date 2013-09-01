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
    return STAT_NOT_IMPLEMENTED_YET;
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
        if (i % pGlobals->settings->circuit.genomeWidth == pGlobals->settings->circuit.genomeWidth - 1) {
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
    ofstream file(fileName);
    if (!file.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot write genome (" << fileName << ")." << endl;
        return STAT_FILE_WRITE_FAIL;
    }
    // output file header with current circuit configuration
    file << pGlobals->settings->circuit.numLayers << " \t(number of layers)" << endl;
    file << pGlobals->settings->circuit.sizeLayer << " \t(size of inside layer)" << endl;
    file << pGlobals->settings->circuit.sizeInput << " \t(number of inputs, without memory)" << endl;
    file << pGlobals->settings->circuit.sizeOutput << " \t(number of outputs, without memory)" << endl;
    file << pGlobals->settings->circuit.sizeMemory << " \t(size of memory)" << endl;
    file << pGlobals->settings->circuit.numConnectors << " \t(maximum number of inside connectors)" << endl;
    file << endl;
    // output circuit itself, starting with input pseudo-layer
    for (int slot = 0; slot < pGlobals->settings->circuit.sizeInputLayer; slot++) {
        file << "IN  [2^" << setw(2) << setfill('0') << slot << "]" << " | ";
    }
    file << endl << endl;
    GENOME_ITEM_TYPE gene;
    int previousLayerWidth;
    int layerWidth;
    int connectorWidth;
    for (int layer = 0; layer < 2 * pGlobals->settings->circuit.numLayers; layer++) {
        previousLayerWidth = layer < 2 ? pGlobals->settings->circuit.sizeInputLayer : pGlobals->settings->circuit.sizeLayer;
        layerWidth = layer < pGlobals->settings->circuit.numLayers*2-2 ? pGlobals->settings->circuit.sizeLayer : pGlobals->settings->circuit.sizeOutputLayer;
        connectorWidth = (layer < 2 || layer >= pGlobals->settings->circuit.numLayers*2-2) ? previousLayerWidth : pGlobals->settings->circuit.numConnectors;
        for (int slot = 0; slot < layerWidth; slot++) {
            gene = genome.gene(layer * pGlobals->settings->circuit.genomeWidth + slot);
            if (layer % 2 == 0) { // connector layer
                gene = relativeToAbsoluteConnectorMask(gene, slot, previousLayerWidth, connectorWidth);
                file << setw(10) << setfill('0') << gene << "   ";
            } else { // function layer
                file << setw(4) << setfill(' ') << left << functionToString(nodeGetFunction(gene));
                file << " [" << setw(3) << setfill('0') << (int) nodeGetArgument(gene,1) << "]" << "   ";
            }
        }
        file << endl;
        if (layer % 2 == 1) file << endl;
    }
    file.close();
    return STAT_OK;
}

int CircuitIO::genomeToCode(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, string fileName) {
    return STAT_NOT_IMPLEMENTED_YET;
}

int CircuitIO::genomeToGraph(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, string fileName) {
    int layerWidth;
    int previousLayerWidth;
    int connectorWidth;
    GENOME_ITEM_TYPE connectors;
    int connection;
    ofstream file(fileName);
    if (!file.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot write genome (" << fileName << ")." << endl;
        return STAT_FILE_WRITE_FAIL;
    }
    // graph header
    file << "digraph EACircuit {" << endl << "rankdir=BT;" << endl << "edge [dir=none];" << endl << "size=\"6,6\";" << endl << "ordering=out;" << endl;

    // node specification
    // input nodes
    file << "node [color=green, style=filled];" << endl << "{ rank=same; ";
    for (int slot = 0; slot < pGlobals->settings->circuit.sizeInputLayer; slot++) { file << "\"" << getNodeLabel(genome, -1, slot) << "\"; "; }
    file << "}" << endl;
    // inside nodes
    file << "node [color=lightblue2, style=filled];" << endl;
    for (int layer = 0; layer < pGlobals->settings->circuit.numLayers; layer++) {
        layerWidth = layer == pGlobals->settings->circuit.numLayers-1 ? pGlobals->settings->circuit.sizeOutputLayer : pGlobals->settings->circuit.sizeLayer;
        file << "{ rank=same; ";
        for (int slot = 0; slot < layerWidth; slot++) { file << "\"" << getNodeLabel(genome, layer, slot) << "\"; "; }
        file << "}" << endl;
    }
    // output nodes
    file << "node [color=red];" << endl << "{ rank=same; ";
    for (int slot = 0; slot < pGlobals->settings->circuit.sizeOutputLayer; slot++) { file << "\"" << getNodeLabel(genome, -2, slot) << "\"; "; }
    file << "}" << endl;

    // connectors
    for (int layer = 0; layer < pGlobals->settings->circuit.numLayers + 1; layer++) {
        previousLayerWidth = layer == 0 ? pGlobals->settings->circuit.sizeInputLayer : pGlobals->settings->circuit.sizeLayer;
        layerWidth = layer < pGlobals->settings->circuit.numLayers ? pGlobals->settings->circuit.sizeLayer : pGlobals->settings->circuit.sizeOutputLayer;
        connectorWidth = (layer == 0 || layer >= pGlobals->settings->circuit.numLayers-1) ? previousLayerWidth : pGlobals->settings->circuit.numConnectors;
        for (int slot = 0; slot < layerWidth; slot++) {
            if (layer == pGlobals->settings->circuit.numLayers) { // last pseudo-output layer
                file << "\"" << getNodeLabel(genome, -2, slot) << "\" -> \"";
                file << getNodeLabel(genome, pGlobals->settings->circuit.numLayers-1, slot) << "\";" << endl;
            } else { // common layer
                connectors = genome.gene(layer * 2 * pGlobals->settings->circuit.genomeWidth + slot);
                connectors = relativeToAbsoluteConnectorMask(connectors, slot, previousLayerWidth, connectorWidth);
                while (connectorsDiscartFirst(connectors,connection)) {
                    file << "\"" << getNodeLabel(genome, layer, slot) << "\" -> \"" << getNodeLabel(genome, layer-1, connection) << "\";" << endl;
                }
            }
        }
    }

    // footer & close
    file << "}";
    file.close();
    return STAT_OK;
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
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->circuit.sizeInput).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("size_output_layer");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->circuit.sizeOutput).c_str()));
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("size_memory");
    pElem2->LinkEndChild(new TiXmlText(toString(pGlobals->settings->circuit.sizeMemory).c_str()));
    pElem->LinkEndChild(pElem2);
    pRoot->LinkEndChild(pElem);

    return pRoot;
}
