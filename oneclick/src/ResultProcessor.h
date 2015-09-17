#ifndef RESULTPROCESSOR_H
#define RESULTPROCESSOR_H

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <exception>
#include <stdexcept>
#include <regex>
#include <iostream>
#include <iomanip>

#include "FileSystem.h"
#include "OneclickConstants.h"
#include "PostProcessor.h"
#include "PValuePostPr.h"
#include "AvgValPostPr.h"

class ResultProcessor {
private:
    PostProcessor * pprocessor;
public:
    /** Constructor for ResultProcessor class.
      * Checks given directory for config and log errors.
      * If directory is valid, processes its results and writes them into file.
      * @paran path             path to directory with results
      *                         (one directory per batch)
      * @param pprocNum         constant of post-processor to be     
      * @throw runtime_error    if invalid directory was given
      */
    ResultProcessor(std::string path , int pprocNum);
private:
    /** Checks config files on given paths for differences.
      * @param configPaths      vector of paths to be checked
      * @param algName          empty string, algorithm name will be set into it
      * @param dirLogger        batch specific logger, logs different configs
      * @return                 true if all configs are same, false otherwise
      */
    bool checkConfigs(const std::vector<std::string> & configPaths , std::string & algName , FileLogger * dirLogger);

    /** Checks logs on given paths for errors and warnings.
      *    If log has no errors, results from log are processed and pValue is stored.
      * @param logPaths           paths to logs to process
      * @param pValues            string to store pValues from logs with no errors
      * @dirLogger                logs batch specifis events =
      *                           = errors, warnings, missing results, 
      *                           inconsistent results, too many results
      * @return result            0-1 => uniformLogs/validLogs, if validLogs = 0
      *                           returns error constant (ERROR_NO_VALID_FILES)
      */
    void checkErrorsProcess(const std::vector<std::string> & logPaths , FileLogger * dirLogger);

    /** Recursively searches given directory,
      * stores paths to files with given index.
      * Paths are sorted in ascending order.
      * @param directory         directory to search in
      * @param paths             paths to files
      * @param fileIndex         file indexes to look for
      * @throws runtime_error    if given argument is not a directory
      */
    void getFilePaths(const std::string & directory , std::vector<std::string> & paths , int fileIndex);

    /** Gets all paths to directories in given directory.
      * Files and directories . and .. are ignored.
      * Paths are sorted in ascending order.
      * @param directory            directory to look in
      * @param dirPaths             output vector with dirPaths
      *    @throws runtime_error    if given directory argument isn't
      *                             path to directory
      */
    void getDirectoryPaths(const std::string & directory, std::vector<std::string> & paths);

    /** Sorts strings.
      * @param strings      strings to be sorted
      */
    void sortStrings(std::vector<std::string> & strings);

    /** Retrieve tag <NOTES> from config file.
      * @param config              config file loaded into string
      * @return                    string with notes
      */
    std::string getNotes(std::string config);

    /** Initializes corresponding post-processor.
      * @param pprocNum                constant of requested PP
      * @throws std::runtime_error     if no case for requested constant 
      *                                exists
      */
    void initPProcessor(int pprocNum);
};

#endif //RESULTPROCESSOR_H
