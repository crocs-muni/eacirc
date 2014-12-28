#ifndef ONECLICKCONSTANTS_H
#define ONECLICKCONSTANTS_H

#include <string>
#include <sstream>

#include "XMLProcessor.h"
#include "EACconstants.h"
#include "EstreamInterface.h"
#include "Sha3Interface.h"

/** Constants used in Oneclick app, mostly default values, filepaths
  * and paths to elements in XML file. Common methods declared here.
  * If XML structure changes over time change it in here also!!! 
  * Otherwise Oneclick won't work. Same goes for default values
  * in script samples.
  */

//Default file names
#define FILE_SCRIPT_UPLOAD			"upload_script.pl"
#define FILE_SCRIPT_UPLOAD_SAMPLE	"upload_script_sample.pl"
#define FILE_SCRIPT_DOWNLOAD		"download_script.pl"
#define FILE_SCRIPT_DOWNLOAD_SAMPLE	"download_script_sample.pl"

//Default paths to directories
#define DIRECTORY_CFGS				"./configs/"
#define DIRECTORY_RESULTS			"./results/"
#define DIRECTORY_SCRIPT_SAMPLES	"./script_samples/"

/////////////////////////////////////////////////////
//********Default paths in XML config file.********//
/////////////////////////////////////////////////////

//Paths to tags in Oneclick section
#define PATH_OC_ALGS				"OC_OPTIONS/ALGORITHMS"
#define PATH_OC_RNDS				"OC_OPTIONS/ROUNDS"
#define PATH_OC_SPEC_RNDS			"OC_OPTIONS/SPECIFIC_ROUNDS"
#define PATH_OC_CLONES				"OC_OPTIONS/CLONES"
#define PATH_OC_WU_ID				"OC_OPTIONS/WU_IDENTIFIER"
#define PATH_OC_DELAY_BND			"OC_OPTIONS/DELAY_BOUND"
#define PATH_OC_NUM_GENS			"OC_OPTIONS/NUM_GENERATIONS"

//Paths to tags in EACirc section - same for all projects
#define PATH_EACIRC					"EACIRC"
#define PATH_EAC_GENS				"EACIRC/NUM_GENERATONS"
#define PATH_EAC_PROJECT			"EACIRC/MAIN/PROJECT"
#define PATH_EAC_NOTES				"EACIRC/NOTES"

//Paths specific for eStream project
#define PATH_ESTR_ALG				"EACIRC/ESTREAM/ALGORITHM_1"
#define PATH_ESTR_RND				"EACIRC/ESTREAM/ROUNDS_ALG_1"

//Paths specific for SHA3 project
#define PATH_SHA3_ALG				"EACIRC/SHA3/ALGORITHM_1"
#define PATH_SHA3_RND				"EACIRC/SHA3/ROUNDS_ALG_1"

//Add new project's paths below - path to algorithm and round tags have to be specified!!



/** Following constants represents keywords used in sript samples.
  * Keywords in generated scripts are replaced by values parsed 
  * from config XML file. In case of changing script structure 
  * change it here accordingly. Methods for script manipulation
  * are located in FileGenerator class.
  */

/////////////////////////////////////////////////////
//*****Keywords and methods in script samples******//
/////////////////////////////////////////////////////

#define KEYWORD_WU_NAME					"WU_NAME"
#define KEYWORD_CONFIG_PATH				"CONFIG_PATH"
#define KEYWORD_CLONES					"CLONE_COUNT"
#define KEYWORD_DELAY_BOUND				"DELAY_BOUND"
#define KEYWORD_DIRECTORY_PATH			"DIRECTORY_PATH"
#define KEYWORD_WU_NAME					"WU_NAME"
#define KEYWORD_WU_DIRECTORY			"WU_DIRECTORY"
#define KEYWORD_METHOD_CREATE_DIRECTORY	"CREATE_DIRECTORY"
#define KEYWORD_METHOD_DOWNLOAD_CLONES	"DOWNLOAD_CLONES"
#define KEYWORD_METHOD_CREATE_WU		"CREATE_WU"

/////////////////////////////////////////////////////
//*****Default values used in script samples*******//
/////////////////////////////////////////////////////

#define DEFAULT_METHOD_CREATE_WU_NAME			"create_wu"			//name of method for creation of single worunit in script
#define DEFAULT_METHOD_CREATE_DIRECTORY_NAME	"create_directory"	//method for creation of specified folder
#define DEFAULT_METHOD_DOWNLOAD_CLONES_NAME		"download_clones"	//method for downloading all clones of single alg from BOINC server
#define DEFAULT_SCRIPT_LINE_SEPARATOR			";"					//separator of line in scripts (should be changed only for good reason)

/////////////////////////////////////////////////////
//*************Global methods**********************//
/////////////////////////////////////////////////////

#define min(a,b) (((a)<(b))?(a):(b))

/** Method used for getting human-readable names of projects and algorithms.
  * Also sets algorithm and number of rounds into given config file.
  * When new project is added to EACirc framework, new "case" have to be added for project in this method!!!
  * Case sets projectName and algorithmName to human-readable destription of project and algorithm.
  * Case also sets values of ALGORITHM and ROUND in project specific settings. Paths to these
  * tags should be added into Oneclick constants.
  * @param root					root of DOM structure of XML config
  * @param projectConstant		constant of project
  * @param algorithmConstant	constant of algorithm
  * @paran rounds				number of rounds used
  * @param projectName			name of project will be entered here
  * @param algorithmName		name of algorithm will be entered here
  * @throws						throws runtime_error in case that project or algorithm constant
  *								doesn't refer to any existing project/alg
  */
void setAlgorithmSpecifics(TiXmlNode * root , int projectConstant , int algorithmConstant , int rounds , std::string * projectName , std::string * algorithmName);

/** Converts integral value to string
  * @param x		integer to be converted
  * @return	string	converted integer		
  */
std::string itostr(int x);

#endif //ONECLICKCONSTANTS_H