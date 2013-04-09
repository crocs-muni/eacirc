#include "Main.h"
#include "EACirc.h"
#include <thread>
//#include <unistd.h> // unistd is not in VS compiler!

#define CATCH_CONFIG_RUNNER
#include "self_tests/Catch.h"

int main(int argc, char **argv) {
    // used in debug mode, wait for debugger to attach
	std::this_thread::sleep_for(std::chrono::milliseconds(3));

    string configFilename = FILE_CONFIG;
    // COMMAND LINE ARGUMENTS PROCESSING
    int argument = 0;
    while (argument + 1 < argc) {
        argument++;
        // RUN SELF-TESTS
        if (strcmp(argv[argument],CMD_OPT_SELF_TEST) == 0) {
            return Catch::Main(argc-argument,argv+argument);
        }
        // CUSTOM CONFIG FILE
        if (strcmp(argv[argument],CMD_OPT_CUSTOM_CONFIG) == 0) {
            if (argument+1 == argc) {
                mainLogger.out(LOGGER_ERROR) << "Incorrect CLI arguments: empty name of custom config file." << endl;
                return STAT_INVALID_ARGUMETS;
            } else {
                configFilename = argv[argument+1];
                argument++;
            }
            continue;
        }
        // LOGGING TO CLOG
        if (strcmp(argv[argument],CMD_OPT_LOGGING) == 0) {
            mainLogger.setOutputStream();
            mainLogger.setlogging(true);
            continue;
        }
        // LOGGING TO FILE
        if (strcmp(argv[argument],CMD_OPT_LOGGING_TO_FILE) == 0) {
            mainLogger.setOutputFile();
            mainLogger.setlogging(true);
            continue;
        }
        // INCORRECT CLI OPTION
        mainLogger.out(LOGGER_ERROR) << "\"" << argv[argument] << "\" is not a valid argument." << endl;
        mainLogger.out() << "Only valid arguments for EACirc are:" << endl;
        mainLogger.out() << "  " << CMD_OPT_LOGGING << "  (set logging to clog)" << endl;
        mainLogger.out() << "  " << CMD_OPT_LOGGING_TO_FILE << "  (set logging to logfile)" << endl;
        mainLogger.out() << "  " << CMD_OPT_SELF_TEST << "  (run self-tests, use " << CMD_OPT_SELF_TEST << " -h to display options)" << endl;
        mainLogger.out() << "  " << CMD_OPT_CUSTOM_CONFIG << " <filename>  (use custom configuration file)" << endl;
        return STAT_INVALID_ARGUMETS;
    }

    EACirc eacirc;
    eacirc.loadConfiguration(configFilename);
    eacirc.prepare();
    eacirc.initializeState();
    eacirc.run();

    if (eacirc.getStatus() != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "EACirc run failed." << endl;
        mainLogger.out() << "       status: " << statusToString(eacirc.getStatus()) << endl;
    } else {
        mainLogger.out(LOGGER_INFO) << "EACirc run succeeded." << endl;
    }
}
