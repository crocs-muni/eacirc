#include "CheckerMain.h"
#include "Checker.h"

int main(int argc, char **argv) {
    // TBD process CLI arguments 

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
