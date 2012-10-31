#ifndef XMLPROCESSOR_H
#define XMLPROCESSOR_H

//libinclude (tinyXML/tinyxml.h)
#include "tinyxml.h"
#include "EACglobals.h"

int LoadConfigScript(string filePath, BASIC_INIT_DATA* pBasicSettings);

int saveXMLFile(TiXmlNode* pRoot, string filename);

/** gets root element from XML file
  * data structure reference by this method is allocated dynamicly
  * CALLER is responsible for freeing!
  *
  * @param pRoot   parsed XML structure will put here
  *                initially should be NULL, otherwise memory leak occurs
  * @param filename
  */
int loadXMLFile(TiXmlNode*& pRoot, string filename);

// also implemented in XMLProcessor.cpp:
//
// QunatumRndGen::QunatumRndGen(const TiXmlHandle root);
// TiXmlHandle QunatumRndGen::exportGenerator();
//
// BiasRndGen::BiasRndGen(const TiXmlHandle root);
// TiXmlHandle BiasRndGen::exportGenerator();

#endif // XMLPROCESSOR_H
