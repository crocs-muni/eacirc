#ifndef ONECLICKCONSTANTS_H
#define ONECLICKCONSTANTS_H

#include <string>
#include <sstream>

#include "EACconstants.h"
#include "estream/EstreamCiphers.h"
#include "sha3/Sha3Functions.h"
#include "caesar/CaesarCiphers.h"

#include "XMLproc.h"
#include "FileLogger.hpp"
#include "Utils.h"

/** Constants used in Oneclick app, mostly default values, filepaths
  * and paths to elements in XML file. 
  * If XML structure changes over time change it in here also!!! 
  * Otherwise Oneclick won't work. Same goes for default values
  * in script samples.
  */

/////////////////////////////////////////////////////
//**************Default file names*****************//
/////////////////////////////////////////////////////

/** Effect of changing these values will be newly generated
  * files with different names. Application will look for new names too.
  */
#define FILE_SCRIPT_UPLOAD             "upload_script.pl"
#define FILE_SCRIPT_UPLOAD_SAMPLE      "upload_script_sample.pl"
#define FILE_SCRIPT_DOWNLOAD           "download_script.pl"
#define FILE_SCRIPT_DOWNLOAD_SAMPLE    "download_script_sample.pl"
#define FILE_LOG                       "oneclick.log"
#define FILE_PROCESSED_RESULTS         "processed_results.txt"

/////////////////////////////////////////////////////
//**************Default file indexes***************//
/////////////////////////////////////////////////////

/** At the moment only configs and logs are used, can change in time
  * Index is at the beginning of file name, separated by INDEX_SEPARATOR
  * INDEX_SEPARATOR constant is defined in Utils.h file
  */
#define INDEX_CONFIG                   0
#define INDEX_EACIRC                   1
#define INDEX_SCORES                   2
#define INDEX_FITNESS_PROGRESS         3
#define INDEX_HISTOGRAMS               4
#define INDEX_POPULATION_INITIAL       5
#define INDEX_POPULATION               6
#define INDEX_STATE_INITIAL            7
#define INDEX_STATE                    8
#define INDEX_EAC_CIRCUIT_XML          9
#define INDEX_EAC_CIRCUIT_DOT         10
#define INDEX_EAC_CIRCUIT_C           11
#define INDEX_EAC_CIRCUIT_TXT         12

/////////////////////////////////////////////////////
//*********Constants of program modes**************//
/////////////////////////////////////////////////////

//Add new modes at the end, not beginning (next should be 3)
#define MODE_FILE_GENERATION          1
#define MODE_RESULT_PROCESSING        2

/////////////////////////////////////////////////////
//********Constants of post processors*************//
/////////////////////////////////////////////////////

//Add new postprocessors at the end, not beginning (next should be 3)
#define PPROCESSOR_PVAL                1
#define PPROCESSOR_AVG                 2

/////////////////////////////////////////////////////
//********Default paths to directories.************//
/////////////////////////////////////////////////////

/** Directory path have to end with a separator!!
  * Use "/" instead of "\" as a separator, Windows too
  */
#define DIRECTORY_CFGS                "./configs/"
#define DIRECTORY_SCRIPT_SAMPLES      "./script_samples/"

/////////////////////////////////////////////////////
//********Default paths in XML config file.********//
/////////////////////////////////////////////////////

//Paths to tags in Oneclick section and attributes
#define PATH_OC_ALGS                "OC_OPTIONS/ALGORITHMS"
#define PATH_OC_RNDS                "OC_OPTIONS/ROUNDS"
#define PATH_OC_SPEC_RNDS           "OC_OPTIONS/SPECIFIC_ROUNDS"
#define PATH_OC_ATT_SPEC_RNDS       "algorithm"
#define PATH_OC_ADD_SETT            "OC_OPTIONS/ADDITIONAL_SETTINGS"
#define PATH_OC_ATT_ADD_SETT        "path"
#define PATH_OC_CLONES              "OC_OPTIONS/CLONES"
#define PATH_OC_WU_ID               "OC_OPTIONS/WU_IDENTIFIER"
#define PATH_OC_NUM_GENS            "OC_OPTIONS/NUM_GENERATIONS"
#define PATH_OC_BOINC_PROJECT       "OC_OPTIONS/BOINC_PROJECT"

//Paths to tags in EACirc section - same for all projects
#define PATH_EACIRC                    "EACIRC"
#define PATH_EAC_PROJECT               "EACIRC/MAIN/PROJECT"
#define PATH_EAC_NOTES                 "EACIRC/NOTES"

/////////////////////////////////////////////////////
//*******Project specific constants****************//
/////////////////////////////////////////////////////

//Paths specific for eStream project + logical name of the project
#define EACIRC_PROJECT_NAME_ESTREAM    "eStream"
#define PATH_ESTR_ALG                  "EACIRC/ESTREAM/ALGORITHM_1"
#define PATH_ESTR_RND                  "EACIRC/ESTREAM/ROUNDS_ALG_1"

//Paths specific for SHA3 project + logical name of the project
#define EACIRC_PROJECT_NAME_SHA3     "SHA3"
#define PATH_SHA3_ALG                "EACIRC/SHA3/ALGORITHM_1"
#define PATH_SHA3_RND                "EACIRC/SHA3/ROUNDS_ALG_1"

//Paths specific for CAESAR project + logical name of the project
#define EACIRC_PROJECT_NAME_CAESAR   "CAESAR"
#define PATH_CAESAR_ALG              "EACIRC/CAESAR/ALGORITHM"
#define PATH_CAESAR_RND              "EACIRC/CAESAR/ALGORITHM_ROUNDS"

//Paths specific for CAESAR project + logical name of the project
#define EACIRC_PROJECT_NAME_FILES    "file distinguisher"

//Add new project's paths below - path to algorithm and round tags have to be specified!!

/////////////////////////////////////////////////////
//****************Scripts constants****************//
/////////////////////////////////////////////////////

/** Following constants represents keywords used in sript samples.
  * Keywords in generated scripts are replaced by values parsed 
  * from config XML file. In case of changing script structure 
  * change it here accordingly. Methods for script manipulation
  * are located in FileGenerator class.
  */

/////////////////////////////////////////////////////
//*****Keywords and methods in script samples.*****//
/////////////////////////////////////////////////////

//common keywords
#define KEYWORD_PROJECT_ID                        "PROJECT_ID_KW"

//upload script keywords
#define KEYWORD_WU_NAME                            "WU_NAME_KW"
#define KEYWORD_CONFIG_PATH                        "CONFIG_PATH_KW"
#define KEYWORD_CLONES                             "CLONES_KW"
#define KEYWORD_METHOD_CREATE_WU                   "CREATE_WU_KW"

//download script keywords
#define KEYWORD_REM_DIR_NAME                       "REM_DIR_NAME_KW"
#define KEYWORD_ARCHIVE_NAME                       "ARCHIVE_NAME_KW"
#define KEYWORD_METHOD_DOWNLOAD_REM_DIR            "DOWNLOAD_REM_DIR_KW"
#define KEYWORD_METHOD_EXTRACT_DELETE_ARCHIVE      "EXTRACT_DELETE_ARCHIVE_KW"


/////////////////////////////////////////////////////
//*****Default values used in script samples.******//
/////////////////////////////////////////////////////

#define DEFAULT_METHOD_CREATE_WU_NAME                 "create_wu"                    //name of method for creation of single worunit in script
#define DEFAULT_METHOD_DOWNLOAD_REM_DIR_NAME          "download_rem_dir"             //name of method for downloading remote directory from BOINC server
#define DEFAULT_METHOD_EXTRACT_DELETE_ARCHIVE_NAME    "extract_delete_archive"       //name of method for extracting and deleting given archive
#define DEFAULT_SCRIPT_LINE_SEPARATOR                 ";"                            //separator of line in scripts (should be changed only for good reason)
                                                                                     //used for detecting end of command

/////////////////////////////////////////////////////
//**************BOINC Project IDs******************//
/////////////////////////////////////////////////////

/** Constants that will be set to scripts to indicate for which project
  * jobs are generated. It's possible that over time, IDs or project names
  * will change. Make relevant changes/additions here and in global method
  * getBoincProjectID!!
  */
#define BOINC_PROJECT_ID_EACIRC_MAIN                   11
#define BOINC_PROJECT_NAME_EACIRC_MAIN                 "EACirc"
#define BOINC_PROJECT_SHORT_EACIRC_MAIN                "EAC"
#define BOINC_PROJECT_ID_EACIRC_CUDA                   3
#define BOINC_PROJECT_NAME_EACIRC_CUDA                 "EACirc_cuda"
#define BOINC_PROJECT_SHORT_EACIRC_CUDA                "EACcuda"
#define BOINC_PROJECT_ID_EACIRC_DEV                    14
#define BOINC_PROJECT_NAME_EACIRC_DEV                  "EACirc_dev"
#define BOINC_PROJECT_SHORT_EACIRC_DEV                 "EACdev"

/////////////////////////////////////////////////////
//**************Values for BOINC*******************//
/////////////////////////////////////////////////////

#define BOINC_MAX_WU_NAME_LENGTH    45

/////////////////////////////////////////////////////
//****************Error return values**************//
/////////////////////////////////////////////////////

//Used by post-processors. No valid files was processed.
#define ERROR_NO_VALID_FILES                    2

/////////////////////////////////////////////////////
//****************Global methods*******************//
/////////////////////////////////////////////////////

class OneclickConstants {
public:

    /** Method used for getting human-readable names of projects and algorithms.
      * Also sets algorithm and number of rounds into given config file.
      * When new project is added to EACirc framework, new "case" have to be added for project in this method!!!
      * Case sets projectName and algorithmName to human-readable destription of project and algorithm.
      * Function uses functions getProjectAlgorithmPath and getProjectRoundPath to set algorithm constant
      * and rounds to project specific paths. When adding new project, add cases to these functions.
      * @param root                    root of DOM structure of XML config
      * @param projectConstant         constant of project
      * @param algorithmConstant       constant of algorithm
      * @paran rounds                  number of rounds used
      * @param projectName             name of project will be entered here
      * @param algorithmName           name of algorithm will be entered here
      * @throws                        throws runtime_error in case that project or algorithm constant
      *                                doesn't refer to any existing project/alg
      */
    static void setAlgorithmSpecifics(TiXmlNode * root , int projectConstant , int algorithmConstant , 
        int rounds , std::string & projectName , std::string & algorithmName);

    /** Returns XML path to project algorithm setting.
      * UPDATE THIS FUNCTION WHEN ADDING NEW PROJECT!!!
      * @param projectConstant          constant of the project
      * @return                         path to algorithm setting
      * @throws                         std::runtime_error if project constant is invalid
      */
    static std::string getProjectAlgorithmPath(int projectConstant);

    /** Returns XML path to project rounds setting.
      * UPDATE THIS FUNCTION WHEN ADDING NEW PROJECT!!!
      * @param projectConstant          constant of the project
      * @return                         path to rounds setting
      * @throws                         std::runtime_error if project constant is invalid
      */
    static std::string getProjectRoundPath(int projetConstant);

    /** Takes logical project name set in Oneclick config and returns project ID
      * belonging to this project. Make changes here should be added new projects 
      * or renamed old constants.
      * @param logicalProjectName          name of project
      * @throws runtime_error              when logicalProjectName is no name of known project
      */
    static int getBoincProjectID(const std::string & logicalProjectName);

    /** Returns shorted project name used in workunit name creation.
      * @param projectID         id of BOINC project
      * @throws runtime_error    if projectID is not a known project ID
      */
    static std::string getBoincProjectShort(int projectID);
};

extern FileLogger oneclickLogger;

#endif //ONECLICKCONSTANTS_H
