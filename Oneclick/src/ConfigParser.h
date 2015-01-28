#ifndef CONFIGPARSER_H
#define CONFIGPARSER_H

#include <iostream>
#include <vector>
#include <string>
#include <exception>

#include "OneclickConstants.h"

class ConfigParser {
private:
	TiXmlNode * root = NULL;
	std::vector<std::vector<int>> algorithmsRounds;
	std::vector<int> numGenerations;
	std::string wuIdentifier;
	int clones;
	int project;
public:
	/** Constructor for ConfigParser class, loads given XML file, parses it,
	  * stores values in variables. Can fail, in that case one of used methods in
	  * constructor will throw runtime_error.
	  * @param path				path to XML file
	  *
	  */
	ConfigParser(std::string path);

	/** Destructor, dealocates root node.
	  */
	~ConfigParser();

	/** Retrieve root of DOM tree.
	  * @return root	root of DOM tree
	  */
	TiXmlNode * getRoot() { return root; };

	/** Returns structure where algorithms with rounds are stored. In case that in
	  * config file was specified invalid algorithm or invalid rounds for algorithm,
	  * these are ignored in generated files.
	  * @return	algorithmsRounds	2D sorted array, each member of array
	  *								represents one algorithm, number of 
	  *								algorithm is first element in array,
	  *								rest are rounds to be used
	  */
	std::vector<std::vector<int>> getAlgorithmsRounds() { return algorithmsRounds; };

	/** Returns vector with sorted counts of generations
	  * to be used in computation.
	  * @return numGenerations		numbers of generations
	  */
	std::vector<int> getNumGenerations() { return numGenerations; };

	/** Returns indetifier of workunits. Can be empty. If used
	  * all generated WU names will begin with same prefix
	  * @return wuIdentifier
	  */
	std::string getWuIdentifier() { return wuIdentifier; };

	/** Returns clones of single workunit. This is a setting for BOINC server. 
	  * @return clones
	  */
	int getClones() { return clones; };

	/** Returns identifier of a project. Used in naming of WUs.
	  * @return project
	  */
	int getProject() { return project; };
private:

	/** Sets class's variable algorithmRounds - representation of for which algs and rounds
	  * should be generated config files.
	  * @param rounds			sorted vector of rounds, 
	  *							will be mapped to values in algorithms
	  * @param algorithm		sorted vector of algorithm constants,
	  *							rounds will be mapped to these algs
	  * @param specificRounds	vector of vectors. Each vector represents one algorithm
	  *							with mapped rounds. First element in vector is alg constant.
	  *							Rounds will NOT be mapped to algs specified in this variable.
	  *							If there are same algs in variables algorithms and specificRounds,
	  *							those in algorithms will be ignored.
	  */
	void setAlgorithmsRounds(std::vector<int> rounds , std::vector<int> algorithms , std::vector<std::vector<int>> specificRounds);

	/** Converts string value from given tag to integer.
	  * Tag must exist and can not be empty
	  * @param path				path to tag with desired value
	  * @return integer
	  * @throws	runtime_error	if in tag are other than numeral values
	  *							if tag doesn't exist/is empty
	  */
	int getXMLValue(std::string path);

	/** Creates vector of sorted integral values without duplicities.
	  * Given tag can be empty.
	  * @param path				path to tag with desired values
	  * @return integers		numeral values from tags
	  */
	std::vector<int> getMultipleXMLValues(std::string path);

	/** Creates vector of vectors, in each vector one algorithm with specified rounds is stored.
	  * First element in each vector is constant of algorithm. Vectors are sorted according to
	  * their first elements. Rounds in vectors are sorted too. From duplicities survive the one latter
	  * in tag. Tag will be ignored if algorithm or rounds are not specified.
	  * @return specificRounds	parsed children of tag SPECIFIC_ROUNDS
	  * @throws runtime_error	if tag ROUNDS has no attribute "algorithm"
	  */
	std::vector<std::vector<int>> getSpecificRounds();

	/** Parses range denoted in tag. Ranges are denoted by dash, in one tag can be multiple ranges.
	  * Integers have to be in ascending order, otherwise range will be ignored. This method is called
	  * from method that is parsing string tag and encounters dash.
	  * @param temp				string representation of bottom border of range
	  * @param elementValue		string that is currently being parsed
	  * @param iterator			position in string being parsed
	  * @param result			vector with integral values
	  * @param path				path to tag currently being parsed
	  * @return					new position in parsed string
	  * @throws	runtime_error	corrupted structure of tag => first parameter is empty,
	  *							space is encoutered right after dash
	  */
	int parseRange(std::string * temp , std::string elementValue , int iterator , std::vector<int> * result , std::string path);

	/** Parses string value into vector of integers. Spaces are separators, dashes denotes ranges.
	  * @param elementValue		value of the string
	  * @return result			vector of integers
	  * @throw runtime_error	invalid characters in tag => other than space, dash or numeral value
	  */
	std::vector<int> parseStringValue(std::string elementValue , std::string path);

	/** Simple insert sort algorithm, sorts vector of integer in ascending order,
	  * after sorting kills duplicities.
	  * @param a				vector to be sorted
	  * @param begin			denotes from which index should start sorting.
	  *							0 sorts whole vector, 1 leaves 1 element unchanged, etc...
	  */
	void sort(std::vector<int> * a , int begin);

	/** Insert sort algoritm, sorts vector of vectors according to their first element.
	  * Kills duplicities => vectors with same first element, the latter one will remain there.
	  * @param a				to be sorted
	  * @return sorted result
	  */
	std::vector<std::vector<int>> sort2D(std::vector<std::vector<int>> a);
};

#endif //CONFIGPARSER_H