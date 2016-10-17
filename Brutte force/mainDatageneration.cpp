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
    ifstream infile(filename.c_str(), ios::binary);
    ifstream::pos_type pos = 1024*1024*1000;//infile.tellg();
    char* result = new char[1024*1024*1000];
    infile.seekg(0, ios::beg);
    infile.read(&result[0], pos);

    return result;
}

int main(int argc, char *argv[]) {



    long num_rounds = 20, Bsize = 1000*1024*1024 ;
    char* buffer = new char[Bsize];

    /*ofstream outfile("RC4Column.bin", ios::binary);
    DataSourceRC4Column S;
    S.read(buffer, Bsize);
    outfile.write(buffer, Bsize);
     */

    string datatypes[] = {"rand","minimalHW","cube1", "cube2", "cube3" };
    string filesnames[] = {"rand128.bin","minimalHW128.bin", "cube1fromrand128.bin", "cube12fromrand128.bin", "cube123fromrand128.bin", "zeroes.bin" };
    string scenario[] = {"keys", "messages", "ivs"};


    char *keys, *messages, *iv;
   // keys = load_array(filesnames[0]);
    messages = load_array(filesnames[1]);
    int messagesize = 16;

    string funcName = "TANGLE";
    //string funcName = "columnAES";
    //DataSourceAES S;
    //DataSourceEstream S(0, ESTREAM_TEA, 64);
    //DataSourceEstream S(0, ESTREAM_DECIM, 6);

    // Keccak needs a special initialization
    //DataSourceSHA3 S(0, SHA3_KECCAK, 32, 32);
    //DataSourceSHA3 S(0, SHA3_MD6, 32, 16);

    //S.read(buffer, Bsize);

    for (int Nr = 1; Nr < 6; ++Nr) {
        //DataSourceAES S(0, Nr);
        //DataSourceSHA3 S(0, SHA3_KECCAK, Nr, 16);
        //DataSourceSHA3 S(0, SHA3_KECCAK, Nr, 32);
        DataSourceSHA3 S(0, SHA3_TANGLE, Nr, 16);
        string fileName = funcName + datatypes[1] + scenario[1]  + "Round" +std::to_string(Nr) + ".bin";
        ofstream outfile(fileName.c_str(), ios::binary);
        S.read(buffer, messages, messagesize, Bsize );
        //S.read(buffer, keys, messages, Bsize );
        outfile.write(buffer, Bsize);
    }

   /* string funcName = "MD6_CRT_";

    for (int Nr = 1; Nr < num_rounds; ++Nr) {
        DataSourceSHA3 S(0,SHA3_MD6,Nr);
        string fileName = funcName + std::to_string(Nr) + ".bin";
        ofstream outfile(fileName.c_str(), ios::binary);
        S.read(buffer, Bsize);
        outfile.write(buffer, Bsize);
    }*/


    return 0;
}
