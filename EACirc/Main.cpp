#include "Main.h"
#include "EACirc.h"
#include <thread>
//#include <unistd.h> // unistd is not in VS compiler!

#define CATCH_CONFIG_RUNNER
#include "self_tests/Catch.h"

int main(int argc, char **argv) {
    //usleep(3000);
	std::this_thread::sleep_for(std::chrono::milliseconds(3));

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
            } else {
            // INCORRECT CLI OPTION
                mainLogger.out() << "\"" << argv[i] << "\" is not a valid argument." << endl;
                mainLogger.out() << "Only valid arguments for EACirc are:" << endl;
                mainLogger.out() << "  " << CMD_OPT_LOGGING << "  (set logging to clog)" << endl;
                mainLogger.out() << "  " << CMD_OPT_LOGGING_TO_FILE << "  (set logging to logfile)" << endl;
                mainLogger.out() << "  " << CMD_OPT_SELF_TEST << "  (run self-tests, use " << CMD_OPT_SELF_TEST << " -h to display options)" << endl;
                return STAT_INVALID_ARGUMETS;
            }
        }
    }

    EACirc eacirc;
    eacirc.loadConfiguration(FILE_CONFIG);
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
