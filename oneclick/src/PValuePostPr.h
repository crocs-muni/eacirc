#ifndef PVALUEPOSTPR_H
#define PVALUEPOSTPR_H

#include <iostream>
#include <sstream>
#include <string>
#include <regex>

#include "OneclickConstants.h"
#include "PostProcessor.h"
#include "FileSystem.h"

#define PVALUES_FILE "pValues.txt"

/** Post-processor to be used with EACirc Categories evaluator (26). Is set as default.
  * Creates file with extracted pValues from logs for each batch.
  * Ratio of uniform workunits in batch is stored in PROCESSED_RESULTS_FILE
  * for each batch.
  */
class PValuePostPr : public PostProcessor {
private:
    int validLogCount;
    int nonUniformLogCount;

    //Stores p values from log files of batch. Dumped to file after every single batch processing
    std::ostringstream pValues;

    std::regex uniPatt;
    std::regex nonUniPatt;
    std::regex pValPatt;

    std::smatch pVal;
    std::smatch emptyMatch;
    std::sregex_token_iterator endExpr;
public:
    PValuePostPr() : validLogCount(0) , nonUniformLogCount(0) ,
        uniPatt("\\[\\d\\d:\\d\\d:\\d\\d\\] info:    KS is not in \\d% interval -> is uniform\\.") ,
        nonUniPatt("\\[\\d\\d:\\d\\d:\\d\\d\\] info:    KS is in \\d% interval -> uniformity hypothesis rejected\\.") ,
        pValPatt("\\[\\d\\d:\\d\\d:\\d\\d\\] info:    KS Statistics: (.*)") {}

    bool process(std::string path) {
        fs::directory_iterator dirIter(path);
        fs::directory_iterator endIter;
        if(dirIter == endIter) throw std::runtime_error("given argument is not a path to existing directory: " + path);
        bool hasValidResults = true;

        for(; dirIter != endIter ; dirIter++) {
            if(dirIter.is_file() && Utils::getFileIndex(dirIter.name()) == INDEX_EACIRC) {
                std::string logFile = Utils::readFileToString(dirIter.path());

                std::sregex_token_iterator uniform(logFile.begin() , logFile.end() , uniPatt , 0);
                std::sregex_token_iterator nonUniform(logFile.begin() , logFile.end() , nonUniPatt , 0);
                std::regex_search(logFile , pVal , pValPatt);

                if(pVal.size() != 2) hasValidResults = false;
                if((uniform != endExpr && nonUniform != endExpr) || (uniform == endExpr && nonUniform == endExpr)) hasValidResults = false;

                if(hasValidResults) {
                    if (uniform != endExpr && nonUniform == endExpr) { validLogCount++; }
                    if (uniform == endExpr && nonUniform != endExpr) { nonUniformLogCount++; validLogCount++; }
                }
                pValues << pVal[1] << std::endl;
                pVal = emptyMatch;
                break;
            }
        }

        return hasValidResults;
    }

    void calculateBatchResults() {
        Score batchScore;
        batchScore.setAlgName(batchName);

        if(validLogCount > 0) {
            batchScore.setVal((float)nonUniformLogCount / (float)validLogCount);
            batchScore.setJobCount(validLogCount);
        } else {
            batchScore.setVal(ERROR_NO_VALID_FILES);
        }
        scores.push_back(batchScore);

        Utils::saveStringToFile(batchDirPath + PVALUES_FILE , pValues.str());
        pValues.str("");

        validLogCount = 0;
        nonUniformLogCount = 0;
        batchDirPath.erase();
        batchName.erase();
    }

    void saveResults() {
        Utils::saveStringToFile(FILE_PROCESSED_RESULTS , writeScores());
        validLogCount = 0;
        nonUniformLogCount = 0;
        batchDirPath.erase();
        batchName.erase();
    }

private:
    std::string writeScores() {
        std::string result("Output file created by p-values result post-processor.\n");
        for(unsigned i = 0 ; i < scores.size() ; i++) {
            result.append(scores[i].toString());
            result.append("\n");
        }
        return result;
    }
};

#endif //PVALUEPOSTPR_H
