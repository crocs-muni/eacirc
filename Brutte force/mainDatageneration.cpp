//
// Created by syso on 9/22/2016.
//

#include <iostream>
#include <fstream>
#include "DataSource/DataSourceAES.h"
#include "DataSource/DataSourceRC4.h"
#include "DataSource/DataSourceRC4Column.h"
#include "DataSource/DataSourceSHA3.h"
#include "DataSource/DataSourceEstream.h"
#include <string>

using namespace std;

//for loading array of keys, messages,..
char* load_array(string filename){
    ifstream infile(filename.c_str(), ios::binary | std::ios::ate);
    std::streamsize size = infile.tellg();
    char* result = new char[size];
    infile.seekg(0, ios::beg);
    infile.read(&result[0], size);

    return result;
}

int main(int argc, char *argv[]) {



    long num_rounds = 20, Bsize = 1000*1024*10 ;
    char* buffer = new char[Bsize];

    string datatypes[] = {"rand","minimalHW","cube1", "cube2", "cube3" };
    string filesnames[] = {"rand128.bin","minimalHW128.bin", "cube1fromrand128.bin", "cube12fromrand128.bin", "cube123fromrand128.bin", "zeroes.bin" };
    string scenario[] = {"keys", "messages", "ivs"};


    char *keys, *messages = NULL, *iv;
    int datatype = 5 ;
   // keys = load_array(filesnames[0]);
    messages = load_array(filesnames[datatype]);
    int messagesize = 32;

    string funcName = "SHA256";
    //string funcName = "TANGLE";
    //string funcName = "columnAES";
    //DataSourceAES S;
    //DataSourceEstream S(0, ESTREAM_TEA, 64);
    //DataSourceEstream S(0, ESTREAM_DECIM, 6);

    // Keccak needs a special initialization
    //DataSourceSHA3 S(0, SHA3_KECCAK, 32, 32);
    //DataSourceSHA3 S(0, SHA3_MD6, 32, 16);

    //S.read(buffer, Bsize);

    for (int Nr = 64; Nr < 65; ++Nr) {
        //DataSourceEstream S(0, ESTREAM_TEA, 64);
        DataSourceSHA3 S(0, SHA3_SHA256, 64, 32, 256);
        //DataSourceAES S(0, Nr);
        //DataSourceSHA3 S(0, SHA3_KECCAK, Nr, 16);
        //DataSourceSHA3 S(0, SHA3_KECCAK, Nr, 32);
        //DataSourceSHA3 S(0, SHA3_TANGLE, Nr, 16);
        //string fileName = funcName + datatypes[datatype] + scenario[1]  + "Round" +std::to_string(Nr) + ".bin";
        string fileName = "Round" +std::to_string(Nr) + ".bin";
        ofstream outfile(fileName.c_str(), ios::binary);

        S.read(buffer, messages, messagesize, Bsize );
        //S.read(buffer, keys, messages, Bsize );
        outfile.write(buffer, Bsize);
        outfile.close();
    }

    return 0;
}
