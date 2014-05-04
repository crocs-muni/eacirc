#ifndef CIRCUITIO_H
#define CIRCUITIO_H

#include "CircuitCommonFunctions.h"
#include "tinyXML/tinyxml.h"
#include "circuit/ReprIO.h"

class CircuitIO : public ReprIO {
public:
    // Non-static wrappers (polymorphic).
    virtual int genomeFromBinary(string binaryCircuit, GAGenome& g);
    virtual int genomeFromText(string filename, GAGenome& g);
    virtual int outputGenomeFiles(GAGenome& g, string fileName = FILE_CIRCUIT_DEFAULT);
    virtual int genomeToBinary(GAGenome& g, string& binaryCircuit);
    virtual int genomeToPopulation(GAGenome& g, string fileName = string(FILE_CIRCUIT_DEFAULT)+".xml");
    virtual int genomeToText(GAGenome& g, string fileName = string(FILE_CIRCUIT_DEFAULT)+".txt");
    virtual int genomeToCode(GAGenome& g, string fileName = string(FILE_CIRCUIT_DEFAULT)+".c");
    virtual int genomeToGraph(GAGenome& g, string fileName = string(FILE_CIRCUIT_DEFAULT)+".dot");
    virtual TiXmlElement* populationHeader(int populationSize);
    
    /** load genome from binary number form
     * @param binaryCircuit (number form in string)
     * @param genome object to fill (by reference)
     * @return status
     */
    static int genomeFromBinarySt(string binaryCircuit, GAGenome& g);

    /** load genome from text format
     * @param filename (file to read genome from)
     * @param genome object to fill (by reference)
     * @return status
     */
    static int genomeFromTextSt(string filename, GAGenome& g);

    /** output genome to external files (TXT, DOT, C, XML)
     * @param genome
     * @param fileName (without suffix)
     * @return status
     */
    static int outputGenomeFilesSt(GAGenome& g, string fileName = FILE_CIRCUIT_DEFAULT);

    /** save genome in number format
     * - no connector transformation
     * - for xml population files
     * @param genome to save
     * @param circuit in number format by reference
     * @return status
     */
    static int genomeToBinarySt(GAGenome& g, string& binaryCircuit);

    /** save genome as population to external file
     * - no connector transformation
     * - for direct xml population loading
     * @param genome to save
     * @param fileName to use, including suffix (FILE_POPULATION if left empty)
     * @return status
     */
    static int genomeToPopulationSt(GAGenome& g, string fileName = string(FILE_CIRCUIT_DEFAULT)+".xml");

    /** save genome in text format to external file
     * - connector transformation applies (saved with absolute connectors)
     * @param genome to save
     * @param fileName to use, including suffix (FILE_POPULATION if left empty)
     * @return status
     */
    static int genomeToTextSt(GAGenome& g, string fileName = string(FILE_CIRCUIT_DEFAULT)+".txt");

    /** save genome as C program to external file
     * @param genome to save
     * @param fileName to use, including suffix (FILE_POPULATION if left empty)
     * @return status
     */
    static int genomeToCodeSt(GAGenome& g, string fileName = string(FILE_CIRCUIT_DEFAULT)+".c");

    /** save genome in graph format (DOT) to external file
     * - use Graphviz to view the file
     * @param genome to save
     * @param fileName to use, including suffix (FILE_POPULATION if left empty)
     * @return status
     */
    static int genomeToGraphSt(GAGenome& g, string fileName = string(FILE_CIRCUIT_DEFAULT)+".dot");

    /** allocate XML structure for header in population file
      * @param populationSize       size of the population (info in the header)
      * @return pointer to root element "eacirc_population"
      */
    static TiXmlElement* populationHeaderSt(int populationSize);
};

#endif // CIRCUITIO_H
