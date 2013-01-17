#include "EstreamProject.h"

EstreamProject::EstreamProject()
    : IProject(PROJECT_ESTREAM) {}

EstreamProject::~EstreamProject() {}

string EstreamProject::shortDescription() const {
    return "eStream candidate ciphers";
}

int EstreamProject::generateTestVectors() {}
