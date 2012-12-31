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
    string loaded;
    REQUIRE(getXMLElementValue(pRoot,location,loaded) == STAT_OK);
    CHECK(loaded == "5.0");

    string newData = "new data here!";
    REQUIRE(setXMLElementValue(pRoot,location,newData) == STAT_OK);
    string attrName = "TEST_ATTR";
    string attrValue = "1234";
    REQUIRE(setXMLElementValue(pRoot,location + "/@" + attrName,attrValue) == STAT_OK);
    REQUIRE(saveXMLFile(pRoot,FILE_CONFIG) == STAT_OK);

    REQUIRE(loadXMLFile(pRoot,FILE_CONFIG) == STAT_OK);
    REQUIRE(getXMLElementValue(pRoot,location,loaded) == STAT_OK);
    CHECK(loaded == newData);
    REQUIRE(getXMLElementValue(pRoot,location+"/@"+attrName,loaded) == STAT_OK);
    CHECK(loaded == attrValue);
}
