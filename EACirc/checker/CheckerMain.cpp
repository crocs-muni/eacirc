#include "CheckerMain.h"
#include "Checker.h"
#include <thread>

int main(int argc, char **argv) {
    //usleep(3000);
    std::this_thread::sleep_for(std::chrono::milliseconds(3));

    //
    // COMMAND LINE ARGUMENTS PROCESSING
    //
    if (argc > 1) {
        int i = 0;
        while (++i < argc) {
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
                return STAT_INVALID_ARGUMETS;
            }
        }
    }

    Checker checker;
    checker.setTestVectorFile(FILE_TEST_VECTORS);
    checker.loadTestVectorParameters();
    checker.check();

    if (checker.getStatus() != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "Checker run failed." << endl;
        mainLogger.out() << "       status: " << statusToString(checker.getStatus()) << endl;
    } else {
        mainLogger.out(LOGGER_INFO) << "Checker run succeeded." << endl;
    }
}
