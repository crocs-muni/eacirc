#include "Main.h"
#include "EACirc.h"

// problem with g++ lower than 4.8, temporary solution
#ifndef _GLIBCXX_USE_NANOSLEEP
#define _GLIBCXX_USE_NANOSLEEP
#endif
#ifdef DEBUG
#include <thread>
#endif

#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

int main(int argc, char **argv) {
    try {
    #ifdef DEBUG
        // used in debug mode, wait for debugger to attach
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    #endif

        string configFilename = FILE_CONFIG;
        // COMMAND LINE ARGUMENTS PROCESSING
        int argument = 0;
        while (argument + 1 < argc) {
            argument++;
            // RUN SELF-TESTS
            if (strcmp(argv[argument],CMD_OPT_SELF_TEST) == 0) {
                testEnvironment();
                return Catch::Main(argc-argument,argv+argument);
            }
            // CUSTOM CONFIG FILE
            if (strcmp(argv[argument],CMD_OPT_CUSTOM_CONFIG) == 0) {
                if (argument + 1 == argc) {
                    mainLogger.out(LOGGER_ERROR) << "Incorrect CLI arguments: empty name of custom config file." << endl;
                    return STAT_INVALID_ARGUMETS;
                } else {
                    configFilename = argv[argument+1];
                    argument++;
                }
                continue;
            }
            // LOGGING TO FILE
            if (strcmp(argv[argument],CMD_OPT_LOGGING_TO_FILE) == 0) {
                if (argument + 1 == argc) {
                    mainLogger.out(LOGGER_ERROR) << "Incorrect CLI arguments: logfile name not provided." << endl;
                    return STAT_INVALID_ARGUMETS;
                }
                mainLogger.setOutputFile(argv[argument+1]);
                argument++;
                continue;
            }
            // INCORRECT CLI OPTION
            mainLogger.out(LOGGER_ERROR) << "\"" << argv[argument] << "\" is not a valid argument." << endl;
            mainLogger.out() << "Only valid arguments for EACirc are:" << endl;
            mainLogger.out() << "  " << CMD_OPT_LOGGING_TO_FILE << " <filename> (set logging to a specific file)" << endl;
            mainLogger.out() << "  " << CMD_OPT_SELF_TEST << "  (run self-tests, use " << CMD_OPT_SELF_TEST << " -h to display options)" << endl;
            mainLogger.out() << "  " << CMD_OPT_CUSTOM_CONFIG << " <filename>  (use custom configuration file)" << endl;
            return STAT_INVALID_ARGUMETS;
        }

        testEnvironment();

        EACirc eacirc;
        eacirc.loadConfiguration( configFilename );
        eacirc.prepare();
        eacirc.initializeState();
        eacirc.run();
        
        // TODO: remove this, when we start using exceptions
        if (eacirc.getStatus() != STAT_OK) {
            mainLogger.out(LOGGER_ERROR) << "EACirc run failed." << endl;
            mainLogger.out() << "       status: " << statusToString(eacirc.getStatus()) << endl;
        } else {
            mainLogger.out(LOGGER_INFO) << "EACirc run succeeded." << endl;
        }
    }
    catch (std::exception& e) {
        mainLogger.out( LOGGER_ERROR ) << "Fatal failure: " << e.what() << std::endl;
        mainLogger.out() << "exiting..." << std::endl;
        return 1;
    }
}

void testEnvironment() {
    if (UCHAR_MAX != 255) {
        mainLogger.out(LOGGER_ERROR) << "Maximum for unsigned char is not 255 (it's " << UCHAR_MAX << ")." << endl;
        exit(-1);
    }
    if (BITS_IN_UCHAR != 8) {
        mainLogger.out(LOGGER_ERROR) << "Unsigned char does not have 8 bits (it has " << BITS_IN_UCHAR << ")." << endl;
        exit(-1);
    }
}
