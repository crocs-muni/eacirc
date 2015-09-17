#ifndef AVGVALPOSTPR_H
#define AVGVALPOSTPR_H

#include <iostream>
#include <string>
#include <sstream>
#include <regex>


#include "OneclickConstants.h"
#include "PostProcessor.h"
#include "FileSystem.h"

#define AVERAGES_FILE "averages.txt"

/** Post-processor to be used with EACirc Top Bit evaluator (25).
  * File with extracted averages from logs is created for every batch.
  * Average of all averages is calculated for every batch, stored in FILE_PROCESSED_RESULTS
  */
class AvgValPostPr : public PostProcessor {
private:
    int validLogCount;
    float avgSum;

    //Stores averages from log files of batch. Dumped to file after every single batch processing
    std::ostringstream averages;

    std::regex avgPatt;

    std::smatch avg;
    std::smatch emptyMatch;
public:
    AvgValPostPr() : validLogCount(0) , avgSum(0) , avgPatt("\\[\\d\\d:\\d\\d:\\d\\d\\] info:    AvgMax: (.*)") {}

    bool process(std::string path) {
        static fs::directory_iterator endIter;
        fs::directory_iterator dirIter(path);
        if(dirIter == endIter) std::runtime_error("given argument is not a path to existing directory: " + path);
        
        bool hasValidResults = true;
        static std::string logFile;

        for(; dirIter != endIter ; dirIter++) {
            if(dirIter.is_file() && Utils::getFileIndex(dirIter.name()) == INDEX_EACIRC) {
                logFile = Utils::readFileToString(dirIter.path());
                std::regex_search(logFile , avg , avgPatt);

                if(avg.size() != 2) {
                    hasValidResults = false;
                } else {
                    float wuAverage = 0;
                    try {
                        wuAverage = std::stof(avg[1] , nullptr);
                        avgSum += wuAverage;
                        validLogCount++;
                        averages << avg[1] << std::endl;
                    } catch(std::invalid_argument e) {
                        hasValidResults = false;
                    } catch(std::out_of_range e) {
                        hasValidResults = false;
                    }
                }

                avg = emptyMatch;
                break;
            }
        }
        return hasValidResults;
    }

    void calculateBatchResults() {
        Score batchScore;
        batchScore.setAlgName(batchName);

        if(validLogCount > 0) {
            batchScore.setVal(avgSum / (float)validLogCount);
            batchScore.setJobCount(validLogCount);
        } else {
            batchScore.setVal(ERROR_NO_VALID_FILES);
        }
        scores.push_back(batchScore);

        Utils::saveStringToFile(batchDirPath + AVERAGES_FILE , averages.str());
        averages.str("");

        validLogCount = 0;
        avgSum = 0;
        batchDirPath.erase();
        batchName.erase();
    }

    void saveResults() {
        Utils::saveStringToFile(FILE_PROCESSED_RESULTS , writeScores());
        
        validLogCount = 0;
        avgSum = 0;
        batchDirPath.erase();
        batchName.erase();
    }

private:
    std::string writeScores() {
        std::string result("Output file created by averages result post-processor.\n");
        for(unsigned i = 0 ; i < scores.size() ; i++) {
            result.append(scores[i].toString());
            result.append("\n");
        }
        return result;
    }
};

#endif //AVGVALPOSTPR_H    
