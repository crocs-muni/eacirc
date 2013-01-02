#ifndef XMLPROCESSOR_H
#define XMLPROCESSOR_H

//libinclude (tinyXML/tinyxml.h)
#include "tinyxml.h"
#include "EACglobals.h"

/** loads configuration data from file to settings structure
  *
  * @param filename         file to read from
  * @param pBasicSettings   pointer to allocated settings structure where to save loaded data
  * @return status
  */
int LoadConfigScript(string filename, BASIC_INIT_DATA* pBasicSettings);

/** saves given XML structure to file
  * automatically add XML declaration
  *
  * @param pRoot        pointer to root element of the structure to be saved
  *                     (this tree structure IS deallocated !!!)
  * @param filename     file to save to (contents overwritten)
  * @return status
  */
int saveXMLFile(TiXmlNode* pRoot, string filename);

/** gets root element from XML file
  * data structure reference by this method is allocated dynamicly
  * CALLER is responsible for freeing!
  *
  * @param pRoot        parsed XML structure will put here
  *                     initially should be NULL, otherwise memory leak occurs
  * @param filename     file where to load from
  * @return status
  */
int loadXMLFile(TiXmlNode*& pRoot, string filename);

/** get string value of XML element or attribute according to path similar to xPath
  * path syntax: "node/node/node" or "node/@attribute"
  *
  * @param pRoot        root of the XML tree
  * @param path         path to requested node/attribute
  * @return value       retrieved node/attribute value (empty string in case of not existing node)
  */
string getXMLElementValue(TiXmlNode*& pRoot, string path);

/** set value of XML element or attribute according to path similar to xPath
  * path syntax: "node/node/node" or "node/@attribute"
  *
  * @param pRoot        root of the XML tree
  * @param path         path to requested node/attribute
  * @param value        new node/attribute value
  * @return status
  */
int setXMLElementValue(TiXmlNode*& pRoot, string path, const string& value);

/** loaclize node in XML tree
  *
  * @param pRoot        root of the XML tree
  * @param path         path to requested node
  * @return pointer to requested node (attribute name in path is ignored)
  *         NULL if given path is invalid
  */
TiXmlNode* getXMLElement(TiXmlNode* pRoot, string path);

#endif // XMLPROCESSOR_H
