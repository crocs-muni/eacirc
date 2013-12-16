#include "CommonFnc.h"
#include "EACglobals.h"

#ifndef M_PI    // to resolve cmath constants problems
#define M_PI 3.141592653589793238462
#endif

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

double chisqr(int Dof, double Cv) {
    if(Cv < 0 || Dof < 1) { return 1; }
    double K = ((double)Dof) * 0.5;
    double X = Cv * 0.5;
    if(Dof == 2) { return exp(-1.0 * X); }
    double gin,gim,gip;
    incog( K, X, gin, gim, gip); // compute incomplete gamma function
    double PValue = gim;
    PValue /= gamma0(K); // divide by gamma function value
    return PValue;
}

int incog(double a,double x,double &gin,double &gim,double &gip) {
    double xam,r,s,ga,t0;
    int k;

    if ((a < 0.0) || (x < 0)) return 1;
    xam = -x+a*log(x);
    if ((xam > 700) || ( a > 170.0)) return 1;
    if (x == 0.0) {
        gin = 0.0;
        gim = gamma0(a);
        gip = 0.0;
        return 0;
    }
    if (x <= 1.0+a) {
        s = 1.0/a;
        r = s;
         for (k=1;k<=60;k++) {
            r *= x/(a+k);
            s += r;
            if (fabs(r/s) < 1e-15) break;
        }
        gin = exp(xam)*s;
        ga = gamma0(a);
        gip = gin/ga;
        gim = ga-gin;
    }
    else {
        t0 = 0.0;
        for (k=60;k>=1;k--) {
            t0 = (k-a)/(1.0+k/(x+t0));
        }
        gim = exp(xam)/(x+t0);
        ga = gamma0(a);
        gin = ga-gim;
        gip = 1.0-gim/ga;
    }
    return 0;
}

/* gamma function.
Algorithms and coefficient values from "Computation of Special
Functions", Zhang and Jin, John Wiley and Sons, 1996.
(C) 2003, C. Bond. All rights reserved.
taken from http://www.crbond.com/math.htm */
double gamma0(double x) {
    int i,k,m;
    double ga,gr,r,z;

    static double g[] = {
        1.0,
        0.5772156649015329,
       -0.6558780715202538,
       -0.420026350340952e-1,
        0.1665386113822915,
       -0.421977345555443e-1,
       -0.9621971527877e-2,
        0.7218943246663e-2,
       -0.11651675918591e-2,
       -0.2152416741149e-3,
        0.1280502823882e-3,
       -0.201348547807e-4,
       -0.12504934821e-5,
        0.1133027232e-5,
       -0.2056338417e-6,
        0.6116095e-8,
        0.50020075e-8,
       -0.11812746e-8,
        0.1043427e-9,
        0.77823e-11,
       -0.36968e-11,
        0.51e-12,
       -0.206e-13,
       -0.54e-14,
        0.14e-14};

    if (x > 171.0) return 1e308;    // This value is an overflow flag.
    if (x == (int)x) {
        if (x > 0.0) {
            ga = 1.0;               // use factorial
            for (i=2;i<x;i++) {
               ga *= i;
            }
         } else ga = 1e308;
     } else {
        if (fabs(x) > 1.0) {
            z = fabs(x);
            m = (int)z;
            r = 1.0;
            for (k=1;k<=m;k++) {
                r *= (z-k);
            }
            z -= m;
        } else z = x;
        gr = g[24];
        for (k=23;k>=0;k--) {
            gr = gr*z+g[k];
        }
        ga = 1.0/(gr*z);
        if (fabs(x) > 1.0) {
            ga *= r;
            if (x < 0.0) {
                ga = -M_PI/(x*ga*sin(M_PI*x));
            }
        }
    }
    return ga;
}
