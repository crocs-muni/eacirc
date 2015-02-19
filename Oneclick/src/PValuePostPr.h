#ifndef PVALUEPOSTPR_H
#define PVALUEPOSTPR_H

#include <iostream>
#include <string>
#include <regex>

#include "OneclickConstants.h"
#include "PostProcessor.h"
#include "FileSystem.h"

#define PVALUES_FILE "pValues.txt"

/** Post-processor to be used with newer evaluator (26). Is set as default.
  * Creates file with extracted pValues from logs for each batch.
  * Ratio of uniform workunits in batch is stored in PROCESSED_RESULTS_FILE
  * for each batch.
  */
class PValuePostPr : public PostProcessor {
private:
	int validLogCount;
	int uniformLogCount;

	std::vector<Score> scores;
	std::string pValues;

	std::regex uniPatt;
	std::regex nonUniPatt;
	std::regex pValPatt;

	std::smatch pVal;
	std::smatch emptyMatch;
	std::sregex_token_iterator endExpr;
public:
	PValuePostPr() : validLogCount(0) , uniformLogCount(0) {
		uniPatt.assign		("\\[\\d\\d:\\d\\d:\\d\\d\\] info:    KS is not in 5% interval -> is uniform\\.");
		nonUniPatt.assign	("\\[\\d\\d:\\d\\d:\\d\\d\\] info:    KS is in 5% interval -> uniformity hypothesis rejected\\.");
		pValPatt.assign		("\\[\\d\\d:\\d\\d:\\d\\d\\] info:    KS Statistics: (.*)");
	}

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
					if(uniform != endExpr && nonUniform == endExpr) { uniformLogCount++ ; validLogCount++; }
					if(uniform == endExpr && nonUniform != endExpr) { validLogCount++; }
				}

				pValues.append(pVal[1]);
				pValues.append("\n");
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
			batchScore.setVal((float)uniformLogCount / (float)validLogCount);
		} else {
			batchScore.setVal(ERROR_NO_VALID_FILES);
		}
		scores.push_back(batchScore);

		Utils::saveStringToFile(batchDirPath + PVALUES_FILE , pValues);

		validLogCount = 0;
		uniformLogCount = 0;
		batchDirPath.erase();
		batchName.erase();
	}

	void saveResults() {
		Utils::saveStringToFile(FILE_PROCESSED_RESULTS , writeScores());
		validLogCount = 0;
		uniformLogCount = 0;
		batchDirPath.erase();
		batchName.erase();
	}

private:
	std::string writeScores() {
		std::string result("Output file created by p-values result post-processor.\n");
		for(int i = 0 ; i < scores.size() ; i++) {
			result.append(scores[i].toString());
			result.append("\n");
		}
		return result;
	}
};

#endif //PVALUEPOSTPR_H