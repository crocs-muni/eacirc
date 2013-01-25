#include "Sha3Project.h"

Sha3Project::Sha3Project()
    : IProject(PROJECT_ESTREAM) {}

Sha3Project::~Sha3Project() {}

string Sha3Project::shortDescription() const {
    return "SHA-3 candidate functions";
}

int Sha3Project::generateTestVectors() {
    return STAT_NOT_IMPLEMENTED_YET;
}
