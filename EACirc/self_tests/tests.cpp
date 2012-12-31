#include "catch.hpp"
#include "EACglobals.h"
#include "XMLProcessor.h"

TEST_CASE("stupid/number equalities", "different numbers are not equal") {
    int number = 5;
    for (int i=1; i<5; i++) {
        CHECK(i !=number);
    }
}

TEST_CASE("xml/xpath","using simple variation of xpath to get/set element and attribute values in XML") {
    string location = "INFO/VERSION";
    TiXmlNode* pRoot = NULL;
    REQUIRE(loadXMLFile(pRoot,FILE_CONFIG) == STAT_OK);
    CHECK(getXMLElementValue(pRoot,location) == "5.0");

    string newData = "new data here!";
    REQUIRE(setXMLElementValue(pRoot,location,newData) == STAT_OK);
    string attrName = "TEST_ATTR";
    string attrValue = "1234";
    REQUIRE(setXMLElementValue(pRoot,location + "/@" + attrName,attrValue) == STAT_OK);
    REQUIRE(saveXMLFile(pRoot,FILE_CONFIG) == STAT_OK);

    REQUIRE(loadXMLFile(pRoot,FILE_CONFIG) == STAT_OK);
    CHECK(getXMLElementValue(pRoot,location) == newData);
    CHECK(getXMLElementValue(pRoot,location+"/@"+attrName) == attrValue);
}
