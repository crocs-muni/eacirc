#ifndef CONFIGPARSER_H
#define CONFIGPARSER_H

#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <exception>
#include <stdexcept>
#include <regex>

#include "OneclickConstants.h"

/** Class for storing optins in configuration files to be generated.
  * Each object MUST have specified algorithm and algorithmRound attribute.
  * settings can be empty. Each Config object represents one configuration file.
  */
class Config {
private:
    //Algorithm constant
    int algorithm;
    //Number of rounds used with algorithm
    int algorithmRound;
    //Additional settings
    std::vector<std::pair<std::string , int>> settings;
public:
    Config(int alg , int algRnd) : algorithm(alg) , algorithmRound(algRnd) {}

    Config(const Config & pc) {
        algorithm = pc.algorithm;
        algorithmRound = pc.algorithmRound;
        settings = pc.settings;
    }

    void addSetting(const std::string & targetPath , int value) {
        std::pair<std::string , int> setting(targetPath , value);
        settings.push_back(setting);
    }

    int getAlgorithm() { return algorithm; }
    int getAlgorithmRound() { return algorithmRound; }
    std::vector<std::pair<std::string , int>> getSettings() { return settings; }
};

/** Class for parsing Oneclick configuration file
  * Stores options for config creation in configs attribute
  * Every Config object represents one configuration file
  */
class ConfigParser {
private:
    TiXmlNode * root = NULL;
    std::vector<int> numGenerations;
    std::string wuIdentifier;
    int clones;
    int project;
    int boincProjectID;
    std::deque<Config> configs;

    //Typedefs for easier orientation. 
    //algorithm_rounds store 
    //    -first: algorithm constant
    //    -second: rounds that will be used with this constant
    //attribute_values store
    //    -first: attribute of the tag the values was taken from
    //    -second: integral representation of values from tag
    //_v are vectors with values described above
    typedef std::pair<int , std::vector<int>>                         algorithm_rounds;
    typedef std::pair<std::string , std::vector<int>>                 attribute_values;
    typedef std::vector<std::pair<int , std::vector<int>>>            algorithm_rounds_v;
    typedef std::vector<std::pair<std::string , std::vector<int>>>    attribute_values_v;
public:
    /** Constructor for ConfigParser class, loads given XML file, parses it,
      * stores values in variables. Can fail, in that case one of used methods in
      * constructor will throw runtime_error.
      * @param path                path to XML file
      */
    ConfigParser(std::string path);

    /** Destructor, dealocates root node.
      */
    ~ConfigParser();

    /** Retrieve root of DOM tree.
      * @return root    root of DOM tree
      */
    TiXmlNode * getRoot() { return root; }

    /** Returns Config objects from which configuration files will be generated.
      * @return configs
      */
    std::deque<Config> getConfigs() { return configs; }

    /** Returns identifier of workunits. Can be empty. If used
      * all generated WU names will end with same suffix.
      * Generated scripts will have this prefix.
      * @return wuIdentifier
      */
    std::string getWuIdentifier() { return wuIdentifier; }

    /** Returns clones of single workunit. This is a setting for BOINC server. 
      * @return clones
      */
    int getClones() { return clones; }

    /** Returns constant of a project. Used in naming of WUs.
      * @return project constant
      */
    int getProject() { return project; }

    /** Returns ID of BOINC project.
      */
    int getBoincProjectID() { return boincProjectID; }
private:

    /** Creates Config objects. First algorithms and rounds are fetched from 
      * Oneclick config file, additional settings are fetched also. Then, from
      * these settings, every possible combination. After this call, configs
      * can be transformed into files in FileGenerator class. 
      */
    void setConfigs();

    /** Creates structure for algorithms and rounds. Each algorithm for which configs will be created,
      * is specified in returned vector. Members of vector are pairs: first is algorithm constant,
      * second vector of integral values => rounds that will be used for this algorithm
      * @return vector of pairs => specified algorithms
      */
    algorithm_rounds_v createAlgorithmsRounds();

    /** Steps over all children of given parent tag (specified by parentPath) and parses values in children
      * tags. Integral values from tag are stored as vector in second element of each pair.
      * Value of attribute given as argument is stored in first element of pair.
      * @param parentPath               path to tag with children
      * @param childAttribute           each child of parentTag must have this attribute, exception will be throw otherwise
      * @return parsed values           each member of returned vector is pair with attribute string value on first place
      *                                 and vector of integral values from child tag on second place
      * @throws    runtime_error        child tag doesn't have childAttribute attribute. 
      */
    attribute_values_v parseChildrenTags(const std::string & parentPath , const std::string & childAttribute);

    /** Converts string value from given tag to integer.
      * Tag must exist and can not be empty
      * @param path                path to tag with desired value
      * @return integer
      * @throws    runtime_error    if in tag are other than numeral values
      *                             if tag doesn't exist/is empty
      */
    int getXMLValue(const std::string & path);

    /** Creates vector of sorted integral values without duplicities.
      * Given tag can be empty. Calls parseStringValue() method
      * @param path                path to tag with desired values
      * @return integers           numeral values from tags
      */
    std::vector<int> getMultipleXMLValues(const std::string & path);

    /** Parses string value into vector of integers. Spaces are separators, dashes denotes ranges.
      * @param elementValue        value of the string
      * @return result             vector of integers
      * @throw runtime_error       invalid characters in tag => other than space, dash or numeral value
    */
    std::vector<int> parseStringValue(const std::string & elementValue , const std::string & path);

    /** Parses range denoted in tag. Ranges are denoted by dash, in one tag can be multiple ranges.
      * Integers have to be in ascending order, otherwise range will be ignored. This method is called
      * from method that is parsing string tag and encounters dash.
      * Fills result with integral values in given range.
      * @param temp                string representation of bottom of range
      * @param elementValue        string that being parsed
      * @param iterator            position in string being parsed
      * @param result              vector with integral values
      * @param path                path to tag currently being parsed
      * @return                    new position in parsed string
      * @throws runtime_error      corrupted structure of tag => first parameter is empty,
      *                            space is encoutered right after dash
      */
    int parseRange(std::string & temp , const std::string & elementValue , unsigned iterator , std::vector<int> & result , const std::string & path);

    /** Simple insert sort algorithm, sorts vector of integer in ascending order,
      * after sorting kills duplicities.
      * @param a                   vector to be sorted
      * @param begin               denotes from which index should start sorting.
      *                            0 sorts whole vector, 1 leaves first element unmoved, etc...
      */
    void sort(std::vector<int> & a , unsigned begin = 0);

    /** Insert sort algoritm. sorts vector of pairs according to first element in each pair.
      * Kills duplicities => pairs with same first element, the latter one will remain there.
      * @param a                to be sorted
      * @return sorted result
      */
    void sort2D(algorithm_rounds_v & a);

    /** Checks workunit identifier for illegal characters and length.
      * Illegal character is everything except A-Z a-z 0-9 () [] - _
      * @throws std::runtime_error          when wu identifier is not valid
      */
    void checkWUIdentifier();
};

#endif //CONFIGPARSER_H
