#include "CircuitIO.h"
#include "XMLProcessor.h"
#include "CommonFnc.h"

int CircuitIO::genomeFromBinarySt(string binaryCircuit, GAGenome& g) {
    GA1DArrayGenome<GENOME_ITEM_TYPE>& genome = dynamic_cast<GA1DArrayGenome<GENOME_ITEM_TYPE>&>(g);
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

int CircuitIO::genomeFromTextSt(string filename, GAGenome& g) {
    return STAT_NOT_IMPLEMENTED_YET;
}

int CircuitIO::outputGenomeFilesSt(GAGenome& genome, string fileName) {
    int status = STAT_OK;
    status = genomeToTextSt(genome, fileName + ".txt");
    if (status != STAT_OK) mainLogger.out(LOGGER_WARNING) << "Problem writing text genome (" << statusToString(status) << ")." << endl;
    status = genomeToPopulationSt(genome, fileName + ".xml");
    if (status != STAT_OK) mainLogger.out(LOGGER_WARNING) << "Problem writing binary genome (" << statusToString(status) << ")." << endl;
    status = genomeToGraphSt(genome, fileName + ".dot");
    if (status != STAT_OK) mainLogger.out(LOGGER_WARNING) << "Problem writing graph genome (" << statusToString(status) << ")." << endl;
    // TODO/TBD: implement code genomes and uncomment
    //status = genomeToCode(genome, fileName + ".c");
    //if (status != STAT_OK) mainLogger.out(LOGGER_WARNING) << "Problem writing code genome (" << statusToString(status) << ")." << endl;
    return STAT_OK;
}

int CircuitIO::genomeToBinarySt(GAGenome& g, string& binaryCircuit) {
    GA1DArrayGenome<GENOME_ITEM_TYPE>& genome = dynamic_cast<GA1DArrayGenome<GENOME_ITEM_TYPE>&>(g);
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

int CircuitIO::genomeToPopulationSt(GAGenome& g, string fileName) {
    
    int status = STAT_OK;
    TiXmlElement* pRoot = populationHeaderSt(1);
    TiXmlElement* pElem = NULL;
    TiXmlElement* pElem2 = NULL;

    pElem = new TiXmlElement("population");
    string textCircuit;
    status = genomeToBinarySt(g, textCircuit);
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

int CircuitIO::genomeToTextSt(GAGenome& g, string fileName) {
    GA1DArrayGenome<GENOME_ITEM_TYPE>& genome = dynamic_cast<GA1DArrayGenome<GENOME_ITEM_TYPE>&>(g);
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
        file << "IN  [2^" << setw(2) << right << setfill('0') << slot << "]" << "   ";
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
                file << setw(10) << right << setfill('0') << gene << "   ";
            } else { // function layer
                file << setw(4) << left << setfill(' ') << functionToString(nodeGetFunction(gene));
                file << " [" << setw(3) << right << setfill('0') << (int) nodeGetArgument(gene,1) << "]" << "   ";
            }
        }
        file << endl;
        if (layer % 2 == 1) file << endl;
    }
    file.close();
    return STAT_OK;
}

int CircuitIO::genomeToCodeSt(GAGenome& g, string fileName) {
    return STAT_NOT_IMPLEMENTED_YET;
}

int CircuitIO::genomeToGraphSt(GAGenome& g, string fileName) {
    GA1DArrayGenome<GENOME_ITEM_TYPE>& genome = dynamic_cast<GA1DArrayGenome<GENOME_ITEM_TYPE>&>(g);
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
    file << "graph EACircuit {" << endl << "rankdir=BT;" << endl << "ranksep=0.75;" << endl << "ordering=out;" << endl;
    file << "splines=polyline;" << endl << "node [style=filled, color=lightblue2];" << endl << endl;

    // node specification
    // input nodes
    file << "{ rank=same;" << endl << "node [color=goldenrod1];" << endl;
    for (int slot = 0; slot < pGlobals->settings->circuit.sizeInputLayer; slot++) {
        if (slot == pGlobals->settings->circuit.sizeMemory) { file << "node [color=chartreuse3];" << endl; }
        file << "\"-1_" << slot << "\"[label=\"" << (slot < pGlobals->settings->circuit.sizeMemory ? "MEM" : "IN") << "\\n" << slot << "\"];" << endl;
    }
    file << "}" << endl;
    // inside nodes
    for (int layer = 0; layer < pGlobals->settings->circuit.numLayers; layer++) {
        layerWidth = layer == pGlobals->settings->circuit.numLayers-1 ? pGlobals->settings->circuit.sizeOutputLayer : pGlobals->settings->circuit.sizeLayer;
        file << "{ rank=same;" << endl;
        for (int slot = 0; slot < layerWidth; slot++) {
            GENOME_ITEM_TYPE gene = genome.gene((layer*2+1) * pGlobals->settings->circuit.genomeWidth + slot);
            file << "\"" << layer << "_" << slot << "\"[label=\"";
            file << functionToString(nodeGetFunction(gene)) << "\\n" << (int) nodeGetArgument(gene,1) << "\"];" << endl;
        }
        file << "}" << endl;
    }
    // output nodes
    file << "{ rank=same;" << endl << "node [color=goldenrod1];" << endl;
    for (int slot = 0; slot < pGlobals->settings->circuit.sizeOutputLayer; slot++) {
        if (slot == pGlobals->settings->circuit.sizeMemory) { file << "node [color=brown2];" << endl; }
        file << "\"-2_" << slot << "\"[label=\"" << (slot < pGlobals->settings->circuit.sizeMemory ? "MEM" : "OUT") << "\\n" << slot << "\"];" << endl;
    }
    file << "}" << endl;

    // invisible connectors (to preserve order in rows)
    file << "edge[style=invis];" << endl;
    for (int layer = -1; layer < pGlobals->settings->circuit.numLayers + 1; layer++) {
        layerWidth = pGlobals->settings->circuit.sizeLayer;
        if (layer == -1) { layerWidth = pGlobals->settings->circuit.sizeInputLayer; }
        if (layer >= pGlobals->settings->circuit.numLayers - 1) { layerWidth = pGlobals->settings->circuit.sizeOutputLayer; }
        file << "\"" << (layer == pGlobals->settings->circuit.numLayers ? -2 : layer) << "_0\"";
        for (int slot = 1; slot < layerWidth; slot++) {
            file << " -- \"" << (layer == pGlobals->settings->circuit.numLayers ? -2 : layer) << "_" << slot << "\"";
        }
        file << ";" << endl;
    }
    file << endl;

    // connectors
    file << "edge[style=solid];" << endl;
    for (int layer = 0; layer < pGlobals->settings->circuit.numLayers + 1; layer++) {
        previousLayerWidth = layer == 0 ? pGlobals->settings->circuit.sizeInputLayer : pGlobals->settings->circuit.sizeLayer;
        layerWidth = layer < pGlobals->settings->circuit.numLayers - 1 ? pGlobals->settings->circuit.sizeLayer : pGlobals->settings->circuit.sizeOutputLayer;
        connectorWidth = (layer == 0 || layer >= pGlobals->settings->circuit.numLayers-1) ? previousLayerWidth : pGlobals->settings->circuit.numConnectors;
        for (int slot = 0; slot < layerWidth; slot++) {
            if (layer == pGlobals->settings->circuit.numLayers) { // last pseudo-output layer
                file << "\"-2_" << slot << "\" -- \"" << pGlobals->settings->circuit.numLayers - 1 << "_" << slot << "\";" << endl;
            } else { // common layer
                connectors = genome.gene(layer * 2 * pGlobals->settings->circuit.genomeWidth + slot);
                connectors = relativeToAbsoluteConnectorMask(connectors, slot, previousLayerWidth, connectorWidth);
                while (connectorsDiscartFirst(connectors,connection)) {
                    file << "\"" << layer << "_" << slot << "\" -- \"" << layer-1 << "_" << connection << "\";" << endl;
                }
            }
        }
    }

    // footer & close
    file << "}";
    file.close();
    return STAT_OK;
}

TiXmlElement* CircuitIO::populationHeaderSt(int populationSize) {
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

int CircuitIO::genomeFromBinary(string binaryCircuit, GAGenome& g){
    return genomeFromBinarySt(binaryCircuit, g);
}

int CircuitIO::genomeFromText(string filename, GAGenome& g){
    return genomeFromTextSt(filename, g);
}

int CircuitIO::genomeToBinary(GAGenome& g, string& binaryCircuit){
    return genomeToBinarySt(g, binaryCircuit);
}

int CircuitIO::genomeToCode(GAGenome& g, string fileName){
    return genomeToCodeSt(g, fileName);
}

int CircuitIO::genomeToGraph(GAGenome& g, string fileName){
    return genomeToGraphSt(g, fileName);
}

int CircuitIO::genomeToPopulation(GAGenome& g, string fileName){
    return genomeToPopulationSt(g, fileName);
}

int CircuitIO::genomeToText(GAGenome& g, string fileName) {
    return genomeToTextSt(g, fileName);
}


int CircuitIO::outputGenomeFiles(GAGenome& g, string fileName){
    return outputGenomeFilesSt(g, fileName);
}

TiXmlElement* CircuitIO::populationHeader(int populationSize){
    return populationHeaderSt(populationSize);
}
