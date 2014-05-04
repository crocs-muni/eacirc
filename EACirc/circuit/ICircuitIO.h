/**
 * @file ICircuitIO.h
 * @author Martin Ukrop, ph4r05
 */

#ifndef ICIRCUITIO_H
#define	ICIRCUITIO_H

#include "EACglobals.h"
#include "GAGenome.h"
#include "tinyXML/tinyxml.h"

class ICircuitIO {
public:
    ICircuitIO();
    ICircuitIO(const ICircuitIO& orig);
    virtual ~ICircuitIO();
    
    /** load genome from binary number form
     * @param binaryCircuit (number form in string)
     * @param genome object to fill (by reference)
     * @return status
     */
    virtual int genomeFromBinary(string binaryCircuit, GAGenome& g)=0;

    /** load genome from text format
     * @param filename (file to read genome from)
     * @param genome object to fill (by reference)
     * @return status
     */
    virtual int genomeFromText(string filename, GAGenome& g)=0;

    /** output genome to external files (TXT, DOT, C, XML)
     * @param genome
     * @param fileName (without suffix)
     * @return status
     */
    virtual int outputGenomeFiles(GAGenome& g, string fileName = FILE_CIRCUIT_DEFAULT)=0;

    /** save genome in number format
     * - no connector transformation
     * - for xml population files
     * @param genome to save
     * @param circuit in number format by reference
     * @return status
     */
    virtual int genomeToBinary(GAGenome& g, string& binaryCircuit)=0;

    /** save genome as population to external file
     * - no connector transformation
     * - for direct xml population loading
     * @param genome to save
     * @param fileName to use, including suffix (FILE_POPULATION if left empty)
     * @return status
     */
    virtual int genomeToPopulation(GAGenome& g, string fileName = string(FILE_CIRCUIT_DEFAULT)+".xml")=0;

    /** save genome in text format to external file
     * - connector transformation applies (saved with absolute connectors)
     * @param genome to save
     * @param fileName to use, including suffix (FILE_POPULATION if left empty)
     * @return status
     */
    virtual int genomeToText(GAGenome& g, string fileName = string(FILE_CIRCUIT_DEFAULT)+".txt")=0;

    /** save genome as C program to external file
     * @param genome to save
     * @param fileName to use, including suffix (FILE_POPULATION if left empty)
     * @return status
     */
    virtual int genomeToCode(GAGenome& g, string fileName = string(FILE_CIRCUIT_DEFAULT)+".c")=0;

    /** save genome in graph format (DOT) to external file
     * - use Graphviz to view the file
     * @param genome to save
     * @param fileName to use, including suffix (FILE_POPULATION if left empty)
     * @return status
     */
    virtual int genomeToGraph(GAGenome& g, string fileName = string(FILE_CIRCUIT_DEFAULT)+".dot")=0;

    /** allocate XML structure for header in population file
      * @param populationSize       size of the population (info in the header)
      * @return pointer to root element "eacirc_population"
      */
    virtual TiXmlElement* populationHeader(int populationSize)=0;
    
private:

};

#endif	/* ICIRCUITIO_H */

