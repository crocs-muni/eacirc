#include "Status.h"

const char* ErrorToString(int error) {
    switch (error) {
    // BASE ERROR STATUS
    case STAT_OK:                       return "STAT_OK";
    case STAT_NOT_IMPLEMENTED_YET:      return "STAT_NOT_IMPLEMENTED_YET";
    case STAT_NOT_ENOUGHT_MEMORY:       return "STAT_NOT_ENOUGHT_MEMORY";
    case STAT_DATA_CORRUPTED:           return "STAT_DATA_CORRUPTED";
    case STAT_DATA_INCORRECT_LENGTH:    return "STAT_DATA_INCORRECT_LENGTH";
    case STAT_CIPHER_INIT_FAIL:         return "STAT_CIPHER_INIT_FAIL";
    case STAT_ENCRYPT_FAIL:             return "STAT_ENCRYPT_FAIL";
    case STAT_DECRYPT_FAIL:             return "STAT_DECRYPT_FAIL";
    case STAT_FILE_OPEN_FAIL:           return "STAT_FILE_OPEN_FAIL";
    case STAT_NOT_ENOUGHT_DATA_TYPE:    return "STAT_NOT_ENOUGHT_DATA_TYPE";
    case STAT_USERDATA_BAD:             return "STAT_USERDATA_BAD";
    case STAT_KEY_LENGTH_BAD:           return "STAT_KEY_LENGTH_BAD";
    case STAT_CONFIG_DATA_READ_FAIL:    return "STAT_CONFIG_DATA_READ_FAIL";
    case STAT_INVALID_ARGUMETS:         return "STAT_INVALID_ARGUMETS";
    case STAT_CONFIG_INCORRECT:         return "STAT_CONFIG_INCORRECT";
    case STAT_INCOMPATIBLE_PARAMETER:   return "STAT_INCOMPATIBLE_PARAMETER";
    case STAT_FILE_WRITE_FAIL:          return "STAT_FILE_WRITE_FAIL";
    case STAT_CONFIG_SCRIPT_INCOMPLETE: return "STAT_CONFIG_SCRIPT_INCOMPLETE";
    case STAT_PROJECT_ERROR:            return "STAT_PROJECT_ERRoR";
    }
    // NO SPECIAL RULE MATCH
    return "'unknown'";
}

