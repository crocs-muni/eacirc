#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

/** Class for storing result of single batch. Used by PostProcessors.
  */
class Score {
private:
    //Algorithm name - usually in format Project: Algorithm - N rounds
    std::string algName;
    //Number of jobs processed 
    int jobCount = 0;
    //Result value
    float val = 0;
public:
    /** Sets attribute algName.
      * @param a    will be algName
      */
    void setAlgName(std::string a) { algName = a; }

    /** Sets attribute val.
      * @param s    will be val
      */
    void setVal(float s) { val = s; }

    /** Sets number of jobs processed in batch.
      * @jc            number of jobs processed in batch
      */
    void setJobCount(int jc) { jobCount = jc; }

    /** Returns val in string format. Fixed 6 decimal points.
      * @return val in string
      */
    std::string valToString() {
        if (val == ERROR_NO_VALID_FILES) return "no valid results";
        std::stringstream valSStr;
        valSStr << std::setprecision(6);
        valSStr << std::fixed;
        valSStr << val;
        return valSStr.str();
    }

    /** Returns formatted string, contains algname  jobCount  val.
      * @return formatted result
      */
    std::string toString() {
        std::stringstream result;
        result << std::setw(60);
        result << std::left;
        result << algName;
        result << std::setw(5);
        result << std::right;
        result << jobCount;
        result << std::setw(10);
        result << std::left;
        result << " jobs";

        result << std::setprecision(6);
        result << std::fixed;
        if(val == ERROR_NO_VALID_FILES) {
            result << "no valid results";
        } else {
            result << val;
        }
        return result.str();
    }
};

/** Interface for result post-processors. Stores results, calculates values,
  * creates result files. Used in class ResultProcessor.
  */
class PostProcessor {
protected:
    std::string batchDirPath;
    std::string batchName;
    
    //Stores results of all processed batches. Is dumped to file at the end of processing
    std::vector<Score> scores;
public:
    virtual ~PostProcessor() {}
    
    /** Sets directory of batch that is being processed.
      * @param path                full path to batch directory
                                   Have to end with directory separator!
      */
    void setBatchDirectoryPath(std::string path) { batchDirPath = path; }

    /** Sets name of batch that is currently processed. This name should be 
      * human-readable and used in result files to identify batch.
      * @param name                name of the batch
      */
    void setBatchName(std::string name) { batchName = name; }

    /** Returns last added batch score. Used for saving batch results into batch logs.
      */
    Score getLastScore() { return scores.back(); }

    /** Add directory to process in current batch. Argument is
      * path to directory with results of single workunit. In there
      * you can process all files you need.
      * @param path                path to directory with wu results
      * @return success            true on success, false if files are not
      *                            in valid format (too many results, etc...)
      */
    virtual bool process(std::string path) = 0;

    /** Called at end of work with single batch, when all wu directories
      * are processed. Should store result from the batch, generate batch-specific files
      * and erase all other results. Same instance will be then used on another batches.
      * Erase batchDirPath and batchName.
      */
    virtual void calculateBatchResults() = 0;

    /** Called at the end of processing of results. All batches are processed,
      * should generate file(s) with global results for all workunits. PostProcessor won't
      * be used after calling this method.
      */
    virtual void saveResults() = 0;
};

#endif //POSTPROCESSOR_H
