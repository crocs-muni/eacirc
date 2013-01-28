#include "Main.h"
#include "EACirc.h"
#include "standalone_testers/TestDistinctorCircuit.h"

#define CATCH_CONFIG_RUNNER
#include "self_tests/Catch.h"

// global variable definitions
Logger mainLogger;
IRndGen* mainGenerator = NULL;
IRndGen* rndGen = NULL;
IRndGen* biasRndGen = NULL;
GLOBALS* pGlobals = NULL;

int main(int argc, char **argv) {
    //
    // COMMAND LINE ARGUMENTS PROCESSING
    //
    if (argc > 1) {
        int i = 0;
        while (++i < argc) {
            // RUN SELF-TESTS
            if (strcmp(argv[i],CMD_OPT_SELF_TEST) == 0) {
                return Catch::Main(argc-i,argv+i);
            }
            // LOGGING TO CLOG
            if (strcmp(argv[i],CMD_OPT_LOGGING) == 0) {
                mainLogger.setOutputStream();
                mainLogger.setlogging(true);
            } else
            // LOGGING TO FILE
            if (strcmp(argv[i],CMD_OPT_LOGGING_TO_FILE) == 0) {
                mainLogger.setOutputFile();
                mainLogger.setlogging(true);
            } else
            // STATIC CIRCUIT
            if (strcmp(argv[i],CMD_OPT_STATIC) == 0) {
                if (argc >= i && strcmp(argv[i+1],CMD_OPT_STATIC_DISTINCTOR) == 0) {
                    mainLogger.out() << "info: Static circuit, distinctor mode." << endl;
                    return testDistinctorCircuit(string(FILE_TEST_DATA_1), string(FILE_TEST_DATA_2));
                } else {
                    mainLogger.out() << "Please specify the second parameter. Supported options:" << endl;
                    mainLogger.out() << "  " << CMD_OPT_STATIC_DISTINCTOR << "  (use the circuit as distinctor)" << endl;
                    return STAT_INVALID_ARGUMETS;
                }
            } else {
            // INCORRECT CLI OPTION
                mainLogger.out() << "\"" << argv[i] << "\" is not a valid argument." << endl;
                mainLogger.out() << "Only valid arguments for EACirc are:" << endl;
                mainLogger.out() << "  " << CMD_OPT_LOGGING << "  (set logging to clog)" << endl;
                mainLogger.out() << "  " << CMD_OPT_LOGGING_TO_FILE << "  (set logging to logfile)" << endl;
                mainLogger.out() << "  " << CMD_OPT_STATIC << "  (run tests on precompiled circuit)" << endl;
                mainLogger.out() << "  " << CMD_OPT_SELF_TEST << "  (run self-tests, use " << CMD_OPT_SELF_TEST << " -h to display options)" << endl;
                return STAT_INVALID_ARGUMETS;
            }
        }
    }

    EACirc eacirc;
    eacirc.loadConfiguration(FILE_CONFIG);
    eacirc.initializeState();
    eacirc.prepare();
    eacirc.run();

    if (eacirc.getStatus() != STAT_OK) {
        mainLogger.out() << "Error: Program run failed." << endl;
        mainLogger.out() << "       status: " << ErrorToString(eacirc.getStatus()) << endl;
    } else {
        mainLogger.out() << "info: Program run succeeded." << endl;
    }
}
