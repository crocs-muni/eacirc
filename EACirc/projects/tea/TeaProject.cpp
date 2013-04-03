#include "TeaProject.h"

TeaProject::TeaProject()
    : IProject(PROJECT_TEA) {}

TeaProject::~TeaProject() {}

string TeaProject::shortDescription() const {
    return "TEA - Tiny Encryption Algorithm";
}

int TeaProject::generateTestVectors() {
    return STAT_NOT_IMPLEMENTED_YET;
}
