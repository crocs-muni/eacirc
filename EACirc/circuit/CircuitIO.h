#ifndef CIRCUITIO_H
#define CIRCUITIO_H

#include "CircuitCommonFunctions.h"
#include "tinyXML/tinyxml.h"

class CircuitIO {
public:
    /** load genome from binary number form
     * @param binaryCircuit (number form in string)
     * @param genome object to fill (by reference)
     * @return status
     */
    static int genomeFromBinary(string binaryCircuit, GA1DArrayGenome<GENOME_ITEM_TYPE>& genome);

    /** load genome from text format
     * @param filename (file to read genome from)
     * @param genome object to fill (by reference)
     * @return status
     */
    static int genomeFromText(string filename, GA1DArrayGenome<GENOME_ITEM_TYPE>& genome);

    /** output genome to external files (TXT, DOT, C, XML)
     * @param genome
     * @param fileName (without suffix)
     * @return status
     */
    static int outputGenomeFiles(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, string fileName = FILE_CIRCUIT_DEFAULT);

    /** save genome in number format
     * - no connector transformation
     * - for xml population files
     * @param genome to save
     * @param circuit in number format by reference
     * @return status
     */
    static int genomeToBinary(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, string& binaryCircuit);

    /** save genome as population to external file
     * - no connector transformation
     * - for direct xml population loading
     * @param genome to save
     * @param fileName to use, including suffix (FILE_POPULATION if left empty)
     * @return status
     */
    static int genomeToPopulation(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, string fileName = string(FILE_CIRCUIT_DEFAULT)+".xml");

    /** save genome in text format to external file
     * - connector transformation applies (saved with absolute connectors)
     * @param genome to save
     * @param fileName to use, including suffix (FILE_POPULATION if left empty)
     * @return status
     */
    static int genomeToText(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, string fileName = string(FILE_CIRCUIT_DEFAULT)+".txt");

    /** save genome as C program to external file
     * @param genome to save
     * @param fileName to use, including suffix (FILE_POPULATION if left empty)
     * @return status
     */
    static int genomeToCode(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, string fileName = string(FILE_CIRCUIT_DEFAULT)+".c");

    /** save genome in graph format (DOT) to external file
     * - use Graphviz to view the file
     * @param genome to save
     * @param fileName to use, including suffix (FILE_POPULATION if left empty)
     * @return status
     */
    static int genomeToGraph(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, string fileName = string(FILE_CIRCUIT_DEFAULT)+".dot");

    /** allocate XML structure for header in population file
      * @param populationSize       size of the population (info in the header)
      * @return pointer to root element "eacirc_population"
      */
    static TiXmlElement* populationHeader(int populationSize);
};

#endif // CIRCUITIO_H
