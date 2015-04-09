#ifndef AVGVALPOSTPR_H
#define AVGVALPOSTPR_H

#include <iostream>
#include <string>
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

	std::vector<Score> scores;
	std::string averages;

    std::regex avgPatt;

	std::smatch avg;
	std::smatch emptyMatch;
public:
    AvgValPostPr() : validLogCount(0) , avgSum(0) , avgPatt("\\[\\d\\d:\\d\\d:\\d\\d\\] info:    AvgMax: (.*)") {}

	bool process(std::string path) {
		fs::directory_iterator dirIter(path);
		fs::directory_iterator endIter;
		if(dirIter == endIter) std::runtime_error("given argument is not a path to existing directory: " + path);
		bool hasValidResults = true;


		for(; dirIter != endIter ; dirIter++) {
			if(dirIter.is_file() && Utils::getFileIndex(dirIter.name()) == INDEX_EACIRC) {
				std::string logFile = Utils::readFileToString(dirIter.path());

				std::regex_search(logFile , avg , avgPatt);

				if(avg.size() != 2) {
					hasValidResults = false;
				} else {
					float wuAverage = 0;
					try {
						wuAverage = std::stof(avg[1] , nullptr);
						avgSum += wuAverage;
						validLogCount++;
						averages.append(avg[1]);
						averages.append("\n");
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
		} else {
			batchScore.setVal(ERROR_NO_VALID_FILES);
		}
		scores.push_back(batchScore);

		Utils::saveStringToFile(batchDirPath + AVERAGES_FILE , averages);

		validLogCount = 0;
		avgSum = 0;
		batchDirPath.erase();
		batchName.erase();
	}

	void saveResults() {
        std::string results = writeScores();
        Utils::saveStringToFile(FILE_PROCESSED_RESULTS , results);
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
