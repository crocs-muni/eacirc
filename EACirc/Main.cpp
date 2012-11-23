#include "Main.h"
#include "EACirc.h"
#include "standalone_testers/TestDistinctorCircuit.h"

Logger mainLogger;

int main(int argc, char **argv) {
    //
    // COMMAND LINE ARGUMENTS PROCESSING
    //
    bool evolutionOff = false;
    if (argc > 1) {
        int i = 0;
        while (++i < argc) {
            if (strcmp(argv[i],CMD_OPT_LOGGING) == 0) {
                mainLogger.setOutputStream();
                mainLogger.setlogging(true);
            } else
            if (strcmp(argv[i],CMD_OPT_LOGGING_TO_FILE) == 0) {
                mainLogger.setOutputFile();
                mainLogger.setlogging(true);
            } else
              // STATIC CIRCUIT ?
            if (strcmp(argv[i],CMD_OPT_STATIC) == 0) {
                if (argc >= i && strcmp(argv[i+1],CMD_OPT_STATIC_DISTINCTOR) == 0) {
                    mainLogger.out() << "Static circuit, distinctor mode." << endl;
                    return testDistinctorCircuit(string(FILE_TEST_DATA_1), string(FILE_TEST_DATA_2));
                } else {
                    mainLogger.out() << "Please specify the second parameter. Supported options:" << endl;
                    mainLogger.out() << "  " << CMD_OPT_STATIC_DISTINCTOR << "  (use the circuit as distinctor)" << endl;
                    return STAT_INVALID_ARGUMETS;
                }
            } else
              // EVOLUTION IS OFF ?
            if (strcmp(argv[i],CMD_OPT_EVOLUTION_OFF) == 0) {
                evolutionOff = true;
                mainLogger.out() << "Evolution turned off." << endl;
            } else {
                mainLogger.out() << "\"" << argv[1] << "\" is not a valid argument." << endl;
                mainLogger.out() << "Only valid arguments for EACirc are:" << endl;
                mainLogger.out() << "  " << CMD_OPT_LOGGING << "  (set logging to clog)" << endl;
                mainLogger.out() << "  " << CMD_OPT_LOGGING_TO_FILE << "  (set logging to logfile)" << endl;
                mainLogger.out() << "  " << CMD_OPT_STATIC << "  (run tests on precompiled circuit)" << endl;
                mainLogger.out() << "  " << CMD_OPT_EVOLUTION_OFF << "  (do not evolve circuits)" << endl;
                return STAT_INVALID_ARGUMETS;
            }
        }
    }

    EACirc eacirc(evolutionOff);
    eacirc.loadConfiguration(FILE_CONFIG);
    eacirc.initializeState();
    eacirc.prepare();
    eacirc.run();
    eacirc.saveState(FILE_STATE);

    if (eacirc.getStatus() != STAT_OK) {
        mainLogger.out() << "Error: Program run failed." << endl;
        mainLogger.out() << "       status: " << ErrorToString(eacirc.getStatus()) << endl;
    }
}
