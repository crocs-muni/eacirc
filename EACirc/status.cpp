#include "status.h"

const char* ErrorToString(int error) {
    switch (error) {
            // BASE ERROR STATUS
        case STAT_OK:                                   return "SUCCESS";   
        case STAT_NOT_IMPLEMENTED_YET:                  return "NOT_IMPLEMENTED_YET";
        case STAT_SESSION_NOT_OPEN:                     return "SESSION_NOT_OPEN";
        case STAT_NOT_ENOUGHT_MEMORY:                   return "NOT_ENOUGHT_MEMORY";
        case STAT_CHANNEL_SECURITY_NOT_ACHIEVED:        return "CHANNEL_SECURITY_NOT_ACHIEVED";
        case STAT_CHANNEL_DESTROY_FAIL:                 return "CHANNEL_DESTROY_FAIL";
        case STAT_CHANNEL_NOT_EXISTS:                   return "CHANNEL_NOT_EXISTS"; 
        case STAT_INCORRECT_HWT:                        return "INCORRECT_HWT";
        case STAT_INCORRECT_SWA:                        return "INCORRECT_SWA";
        case STAT_DATA_CORRUPTED:                       return "DATA_CORRUPTED";
        case STAT_DATA_INCORRECT_LENGTH:                return "DATA_INCORRECT_LENGTH";
        case STAT_KEY_SCHEDULE_FAIL:                    return "KEY_SCHEDULE_FAIL"; 
        case STAT_CIPHER_INIT_FAIL:                     return "CIPHER_INIT_FAIL"; 
        case STAT_ENCRYPT_FAIL:                         return "ENCRYPT_FAIL"; 
        case STAT_DECRYPT_FAIL:                         return "DECRYPT_FAIL"; 
        case STAT_FILE_OPEN_FAIL:                       return "FILE_OPEN_FAIL"; 
        case STAT_SEAUT_FRESHNESS_FAIL:                 return "SEAUT_FRESHNESS_FAIL";
        case STAT_NO_CONNECTION:                        return "NO_CONNECTION";
        case STAT_NOT_ENOUGHT_DATA_TYPE:                return "NOT_ENOUGHT_DATA_TYPE";
        case STAT_LICENCE_TOO_LONG:                     return "STAT_LICENCE_TOO_LONG";      
        case STAT_RESPONSE_DATA_LENGTH_BAD:             return "STAT_RESPONSE_DATA_LENGTH_BAD";
        case STAT_USERDATA_BAD:                         return "STAT_USERDATA_BAD";
        case STAT_KEY_LENGTH_BAD:                       return "STAT_KEY_LENGTH_BAD";
        case STAT_DATA_TOO_LONG:                        return "STAT_DATA_TOO_LONG";
        case STAT_INI_DATA_WRITE_FAIL:                  return "STAT_INI_DATA_WRITE_FAIL";                             
        case STAT_INI_DATA_READ_FAIL:                   return "STAT_INI_DATA_READ_FAIL";
        case STAT_CODING_NOT_BIJECT:                    return "STAT_CODING_NOT_BIJECT";               
        case STAT_CODING_ALREADY_ASSIGNED:              return "STAT_CODING_ALREADY_ASSIGNED"; 
        case STAT_CONFIG_SCRIPT_INCOMPLETE:             return "STAT_CONFIG_SCRIPT_INCOMPLETE";
        case STAT_IOC_CORRUPTED:                        return "STAT_IOC_CORRUPTED";           
        case STAT_LOAD_AUTH_FNC_FAIL:                   return "STAT_LOAD_AUTH_FNC_FAIL";      
        case STAT_LOAD_LIBRARY_FAIL:                    return "STAT_LOAD_LIBRARY_FAIL";       
        case STAT_SCARD_ERROR:                          return "STAT_SCARD_ERROR";             
        case STAT_FULL_PATH_FAIL:                       return "STAT_FULL_PATH_FAIL";
        case STAT_UNKNOWN_SCARD_PROTOCOL:               return "STAT_UNKNOWN_SCARD_PROTOCOL";
        case STAT_OPERATION_CANCELED:                   return "STAT_OPERATION_CANCELED";
        case STAT_STRING_NOT_FOUND:                     return "STAT_STRING_NOT_FOUND";
        case STAT_COORDINATES_EXCEEDS:                  return "STAT_COORDINATES_EXCEEDS";  
        case STAT_NODE_NOT_EXISTS:                      return "STAT_NODE_NOT_EXISTS";
        case STAT_NO_SUCH_NEIGHBOUR:                    return "STAT_NO_SUCH_NEIGHBOUR";  
        case STAT_PORT_DRIVER_INIT_FAIL:                return "STAT_PORT_DRIVER_INIT_FAIL";
        case STAT_SOFT_KILLED:                          return "STAT_SOFT_KILLED";  
        case STAT_KEY_MISSING:                          return "STAT_KEY_MISSING";    
        case STAT_MESSAGE_MISSING:                      return "STAT_MESSAGE_MISSING";
        case STAT_NOT_NEIGHBOUR:                        return "STAT_NOT_NEIGHBOUR";
        case STAT_INVALID_PTR:                          return "STAT_INVALID_PTR";
        case STAT_KEY_TYPE_BAD:                         return "STAT_KEY_TYPE_BAD";  
		}
        // NO SPECIAL RULE MATCH                                            
        return "'unknown'";   
}  

