#ifndef FILEGENERATOR_H
#define FILEGENERATOR_H

#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <stdexcept>
#include <sstream>
#include <fstream>

#include "ConfigParser.h"

class FileGenerator {
private:
    ConfigParser * parser;
public:
    /** Constructor for FileGenerator class. Inits ConfigParser class, then uses parsed
      * info from XML file to generate config files for EACirc and perl scripts to upload
      * and download them to BOINC server.
      * @param path            path to XML config file for Oneclick.
      */
    FileGenerator(std::string path);

    /** Destructor, deallocates ConfigParser class.
      */
    ~FileGenerator();
private:

    /** Generates XML configs specified in ConfigParser class, PERL script for uploading configs
      * to BOINC server and script for downloading results.
      * @throw runtime_error    if files can't be opened or written into
      */
    void generateFiles();

    /** Gets method prototype from string with loaded sample string. 
      * It will match first occurence of methodName and copies rest 
      * of the line(until DEFAULT_SCRIPT_LINE_SEPARATOR is found). 
      * @param source                string with content of sample script
      * @param methodName            typically one of KEYWORD_METHOD_... constants
      * @return                      prototype of the specified method
      * @throws runtime_error        if methodName has no occurence in source
      */
    std::string getMethodPrototype(const std::string & source , const std::string & methodName);

    /** Replaces one string with another in target string.
      * @param target               string in which changes will be made
      * @param replace              substring to be replaced
      * @param instead              with this string
      * @throw runtime_error        if string replace has no occurence in target string
      */
    void replaceInString(std::string & target , const std::string & replace , const std::string & instead);

    /** Inserts method call into script string.
      * If target string contains method prototype,
      * this will be replaced (indicated by firstInsert). 
      * @param target                target string
      * @param methodPrototype       typicaly method prototype
      * @param toInsert              method with completed arguments,
      *                              without keywords
      * @param position              position where argument toInsert
      *                              will be inserted
      * @param firstInsert           if true, methodPrototype will be
      *                              replaced with toInsert, otherwise
      *                              toInsert is inserted at position
      * @return                      new position - at the end of
      *                              inserted string in target string
      */
    int insertIntoScript(std::string & target , const std::string & methodPrototype , std::string & toInsert , int position , bool firstInsert);
};

#endif //FILEGENERATOR_H
