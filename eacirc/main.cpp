#include "EACirc.h"

#ifdef DEBUG
    #include <thread>
#endif

#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

void testEnvironment();

int main(int argc, char **argv) {
    try {
        string configFilename = FILE_CONFIG;
        // COMMAND LINE ARGUMENTS PROCESSING
        for (int arg = 1; arg < argc; ++arg) {
            // RUN SELF-TESTS
            if (strcmp(argv[arg],CMD_OPT_SELF_TEST) == 0) {
                testEnvironment();
                return Catch::Session().run(argc-arg,argv+arg);
            }
            // CUSTOM CONFIG FILE
            if (strcmp(argv[arg],CMD_OPT_CUSTOM_CONFIG) == 0) {
                if (arg + 1 == argc) {
                    mainLogger.out(LOGGER_ERROR) << "Incorrect CLI arguments: empty name of custom config file." << endl;
                    return STAT_INVALID_ARGUMETS;
                } else {
                    configFilename = argv[arg+1];
                    arg++;
                }
                continue;
            }
            // LOGGING TO FILE
            if (strcmp(argv[arg],CMD_OPT_LOGGING_TO_FILE) == 0) {
                if (arg + 1 == argc) {
                    mainLogger.out(LOGGER_ERROR) << "Incorrect CLI arguments: logfile name not provided." << endl;
                    return STAT_INVALID_ARGUMETS;
                }
                mainLogger.setOutputFile(argv[arg+1]);
                arg++;
                continue;
            }
            // INCORRECT CLI OPTION
            mainLogger.out(LOGGER_ERROR) << "\"" << argv[arg] << "\" is not a valid argument." << endl;
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
    catch (...) {
        throw;
    }
}

void testEnvironment() {
    if (UCHAR_MAX != 255) {
        mainLogger.out(LOGGER_ERROR) << "Maximum for unsigned char is not 255 (it's " << UCHAR_MAX << ")." << std::endl;
        exit(-1);
    }
    if (BITS_IN_UCHAR != 8) {
        mainLogger.out(LOGGER_ERROR) << "Unsigned char does not have 8 bits (it has " << BITS_IN_UCHAR << ")." << std::endl;
        exit(-1);
    }
}
