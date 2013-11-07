#include "CommonFnc.h"
#include <string>
#include <cmath>
#include <sstream>
#include <exception>
#include "EACglobals.h"


using namespace std;

int BYTE_ConvertFromHexStringToArray(string hexaString, unsigned char* pArray, unsigned char* pbArrayLen) {
    int     status = STAT_OK;
    unsigned long   arrayLen = *pbArrayLen;
    
    status = BYTE_ConvertFromHexStringToArray(hexaString, pArray, &arrayLen);
    if (arrayLen > 0xFF) status = STAT_NOT_ENOUGHT_DATA_TYPE;
    else *pbArrayLen = (unsigned char) arrayLen;

    return status;
}

int BYTE_ConvertFromHexStringToArray(string hexaString, unsigned char* pArray, unsigned long* pbArrayLen) {
    int					status = STAT_OK;
    unsigned long       pos = 0;
    unsigned long       pos2 = 0;
    string				hexNum;
    unsigned long       num;
    unsigned char*      pTempArray = NULL;
    unsigned long       tempArrayPos = 0;

    // EAT SPACES
    //hexaString.TrimLeft(); hexaString.TrimRight();
	TrimLeadingSpaces(hexaString);
	TrimTrailingSpaces(hexaString);
    hexaString += " ";
    //hexaString.GetLength();
	hexaString.length();

    if (status == STAT_OK) {
        pTempArray = new unsigned char[hexaString.length()];
        memset(pTempArray, 0, hexaString.length());

        pos = pos2 = 0;
		while ((pos = hexaString.find(' ', pos2)) != string::npos) {
            hexNum = hexaString.substr(pos2, pos - pos2);
            //hexNum.TrimLeft(); hexNum.TrimRight();
			TrimLeadingSpaces(hexNum);
			TrimTrailingSpaces(hexNum);
            if (hexNum.length() > 0) {
				
				std::istringstream iss(hexNum);

				if(!(iss>>std::hex>>num)){
                    mainLogger.out(LOGGER_ERROR) << "BYTE_ConvertFromHexStringToArray: Invalid argument!" << endl;
					exit(1);
				}
        
                if (num == 0xFF) pTempArray[tempArrayPos] = 0xFF;
                else pTempArray[tempArrayPos] = (unsigned char) num & 0xFF;
                
                tempArrayPos++;
            }
            pos2 = pos + 1;
        }

        if (tempArrayPos > *pbArrayLen) {
            status = STAT_NOT_ENOUGHT_MEMORY;
        }  
        else {
            memcpy(pArray, pTempArray, tempArrayPos);
        }
        *pbArrayLen = tempArrayPos;

        if (pTempArray) delete[] pTempArray;
    }

    return status;
}

int BYTE_ConvertFromArrayToHexString(unsigned char* pArray, unsigned long pbArrayLen, string* pHexaString) {
    int				status = STAT_OK;
    string			hexNum;
    unsigned long   i;

    *pHexaString = "";
    for (i = 0; i < pbArrayLen; i++) {
        //hexNum.Format("%.2x", pArray[i]);
		ostringstream os1;
		os1 << pArray[i];
        hexNum = os1.str();
		hexNum += " ";

        *pHexaString += hexNum;
    }

    //pHexaString->TrimRight(" ");
	TrimTrailingSpaces(*pHexaString);

    return status;
}

//trimming functions found here: http://codereflect.com/2007/01/31/how-to-trim-leading-or-trailing-spaces-of-string-in-c/
void TrimLeadingSpaces(string& str) {
	// Code for Trim Leading Spaces only
	size_t startpos = str.find_first_not_of(" \t"); // Find the first character position after excluding leading blank spaces
	if( string::npos != startpos ) {
		str = str.substr( startpos );
	}
}

void TrimTrailingSpaces(string& str) {
	// Code for Trim trailing Spaces only
	size_t endpos = str.find_last_not_of(" \t"); // Find the first character position from reverse af
	if( string::npos != endpos ) {
		str = str.substr( 0, endpos+1 );
	}
}

double StringToDouble(string &s, bool failIfLeftoverChars) {
   std::istringstream i(s);
   double x;
   char c;
   // check for right format and leftover characters
   if (!(i >> x) || (failIfLeftoverChars && i.get(c))) {
       mainLogger.out(LOGGER_ERROR) << "StringToDouble(\"" << s << "\".)" << endl;
       //exit(1);
       x = 0;
   }
     
   return x;
}

int copyFile(string source, string destination) {
    ifstream inFile(source, ios_base::binary);
    if (!inFile.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot open file (" << source << ")." << endl;
        return STAT_FILE_OPEN_FAIL;
    }
    ofstream outFile(destination, ios_base::binary | ios_base::trunc);
    if (!outFile.is_open()) {
        mainLogger.out(LOGGER_ERROR) << "Cannot write to file (" << destination << ")." << endl;
        return STAT_FILE_WRITE_FAIL;
    }
    outFile << inFile.rdbuf();
    inFile.close();
    outFile.close();

    return STAT_OK;
}

// from http://www.codeproject.com/Articles/432194/How-to-Calculate-the-Chi-Squared-P-Value
double chisqr(int Dof, double Cv)
{
    if(Cv < 0 || Dof < 1)
    {
        return 1;
    }
    double K = ((double)Dof) * 0.5;
    double X = Cv * 0.5;
    if(Dof == 2)
    {
    return exp(-1.0 * X);
    }

    double PValue = igf(K, X);
    if(isnan(PValue) || isinf(PValue) || PValue <= 1e-8)
    {
        return 1;
    }

    //PValue /= gamma(K);
    PValue /= tgamma(K);

    return (1.0 - PValue);
}

// from http://www.codeproject.com/Articles/432194/How-to-Calculate-the-Chi-Squared-P-Value
double igf(double S, double Z)
{
    if(Z < 0.0)
    {
    return 0.0;
    }
    double Sc = (1.0 / S);
    Sc *= pow(Z, S);
    Sc *= exp(-Z);

    double Sum = 1.0;
    double Nom = 1.0;
    double Denom = 1.0;

    for(int I = 0; I < 200; I++)
    {
    Nom *= Z;
    S++;
    Denom *= S;
    Sum += (Nom / Denom);
    }

    return Sum * Sc;
}
