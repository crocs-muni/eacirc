//
// Created by Dusan Klinec on 16.06.16.
//

#include "mainBi.h"

#include <iostream>
#include "dynamic_bitset.h"
#include "bit_array.h"
#include "Term.h"
#include "bithacks.h"
#include <ctime>
#include <random>
#include <fstream>
#include "CommonFnc.h"
#include "finisher.h"
#include "logger.h"
#include "TermGenerator.h"

using namespace std;

int main(int argc, char *argv[]) {
    const int numTVs = 100000;
    const int tvsize = 16, numVars = 8 * tvsize;
    const int numEpochs = 40;
    const int numBytes = numTVs * tvsize;
    const int deg = 3;
    u8 *TVs = new u8[numBytes];

    ifstream in(argv[1], ios::binary);


    return 0;
}