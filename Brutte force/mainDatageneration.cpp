//
// Created by syso on 9/22/2016.
//

#include <iostream>
#include <fstream>
#include "DataSource/DataSourceAES.h"
#include "DataSource/DataSourceRC4.h"
#include "DataSource/DataSourceRC4Column.h"
#include "DataSource/DataSourceSHA3.h"
#include <string>

using namespace std;

int main(int argc, char *argv[]) {



    long num_rounds = 20, Bsize = 1000*1024*1024 ;
    char* buffer = new char[Bsize];



    /*ofstream outfile("RC4Column.bin", ios::binary);
    DataSourceRC4Column S;
    S.read(buffer, Bsize);
    outfile.write(buffer, Bsize);
     */


    string funcName = "AES_CRT_";
    DataSourceAES S;
    S.read(buffer, Bsize);

    for (int Nr = 1; Nr < num_rounds; ++Nr) {
        DataSourceAES S(0, Nr);
        string fileName = funcName + std::to_string(Nr) + ".bin";
        ofstream outfile(fileName.c_str(), ios::binary);
        S.read(buffer, Bsize);
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
