#ifndef XMLPROCESSOR_H
#define XMLPROCESSOR_H

//libinclude (tinyXML/tinyxml.h)
#include "tinyxml.h"
#include "EACglobals.h"

int LoadConfigScript(string filePath, BASIC_INIT_DATA* pBasicSettings);

int saveXMLFile(TiXmlHandle root, string filename);

// also implemented in XMLProcessor.cpp:
//
// QunatumRndGen::QunatumRndGen(const TiXmlHandle root);
// TiXmlHandle QunatumRndGen::exportGenerator();
//
// BiasRndGen::BiasRndGen(const TiXmlHandle root);
// TiXmlHandle BiasRndGen::exportGenerator();

#endif // XMLPROCESSOR_H
