#ifndef STATUS_H
#define STATUS_H

#define STAT_OK                                 0       // OK
#define STAT_NOT_IMPLEMENTED_YET                1       // REQUESTED BEHAVIOUR NOT IMPLEMENTED YET
#define STAT_SESSION_NOT_OPEN                   2       // SESSION IS NOT OPEN, TRY TO OPEN SESSION BEFORE CALLING THIS FUNCTION
#define STAT_NOT_ENOUGHT_MEMORY                 3       // NOT ENOUGHT MEMORY TO COMPLETE REQUESTED OPERATION, TRY TO INCREASE GIVEN MEMORY BUFFER SIZE
#define STAT_CHANNEL_SECURITY_NOT_ACHIEVED      4       // LOGICAL CHANNEL SECURITY NOT ACHIVED TO REQUESTED LEVEL
#define STAT_CHANNEL_DESTROY_FAIL               5       // FAIL TO DESTROY LOGICAL CHANNEL
#define STAT_CHANNEL_NOT_EXISTS                 6       // LOGICAL CHANNEL NOT EXIST YET, TRY TO CREATE CHANNEL BEFORE CALLING THIS FUNCTION
#define STAT_INCORRECT_HWT                      7       // INCORRECT TARGET HARDWARE TOKEN 
#define STAT_INCORRECT_SWA                      8       // INCORRECT TARGET SOFTWARE AGENT
#define STAT_DATA_CORRUPTED                     10      // GIVEN DATA ARE NOT IN EXPECTED FORMAT   
#define STAT_DATA_INCORRECT_LENGTH              11      // LENGTH OF GIVEN DATA IS INCORRECT
#define STAT_KEY_SCHEDULE_FAIL                  12      // FAIL TO SCHEDULE CIPHER KEY MATERIAL    
#define STAT_CIPHER_INIT_FAIL                   13      // FAIL TO INICIALIZE CIPHER ENGINE
#define STAT_ENCRYPT_FAIL                       14      // FAIL TO COMPLETE ENCRYPTION OPERATION
#define STAT_DECRYPT_FAIL                       15      // FAIL TO COMPLETE DECRYPTION OPERATION
#define STAT_FILE_OPEN_FAIL                     16      // FAIL TO OPEN TARGET FILE
#define STAT_SEAUT_FRESHNESS_FAIL               17      // MESSAGE RECIEVED DURING SEAUT PROTOCOL IS NOT FRESH, ACCIDENTAL DAMAGE OF DATA OR REPLAY ATTACK DETECTED
#define STAT_NO_CONNECTION                      18      // NO EXISTING CONNECTION, TRY TO ESTABILISH CONNECTION BEFORE CALLING THIS FUNCTION
#define STAT_NOT_ENOUGHT_DATA_TYPE              20      // GIVEN DATA VARIABLE IS UNABLE TO CONTAIN RETURN VALUE
#define STAT_LICENCE_TOO_LONG                   21      // LICENCE LENGTH IS BIGGER THAN MAX. ALLOWED
#define STAT_RESPONSE_DATA_LENGTH_BAD           22      // RESPONSE DATA LENGTH DIFFERS FROM EXPECTED 
#define STAT_USERDATA_BAD                       23      // DATA OBTAINED FROM USER INPUT ARE INVALID
#define STAT_KEY_LENGTH_BAD                     24      // KEY LENGTH DIFFERS FROM EXPECTED
#define STAT_DATA_TOO_LONG                      25      // DATA LENGTH IS BIGGER THAN MAX. ALLOWED
#define STAT_CONFIG_DATA_WRITE_FAIL             26      // FAIL TO WRITE DATA INTO CONFIG FILE
#define STAT_CONFIG_DATA_READ_FAIL              27      // FAIL TO READ DATA FROM CONFIG FILE
#define STAT_CODING_NOT_BIJECT                  28
#define STAT_CODING_ALREADY_ASSIGNED            29
#define STAT_CONFIG_SCRIPT_INCOMPLETE           30      // MISSING ITEMS IN CONFIGURATION SCRIPT    
#define STAT_IOC_CORRUPTED                      31      // I/O CODINGS FORMAT INCORRECT
#define STAT_LOAD_AUTH_FNC_FAIL                 32
#define STAT_LOAD_LIBRARY_FAIL                  33
#define STAT_SCARD_ERROR                        34      // SCARD FUNCTIONS ERROR OCCURED. ERROR VALUE SHOULD BE STORED SOMEWHERE IN OBJECT (::GetLastSCardError()) 
#define STAT_FULL_PATH_FAIL                     35      // FAIL TO CREATE FULL FILE PATH
#define STAT_UNKNOWN_SCARD_PROTOCOL             36
#define STAT_OPERATION_CANCELED                 37      // OPERATION WAS CANCELED BY USER
#define STAT_STRING_NOT_FOUND                   38
#define STAT_COORDINATES_EXCEEDS                39      // COORDINATES EXCEEDS INTERNAL BOUNDS  
#define STAT_NODE_NOT_EXISTS                    40
#define STAT_NO_SUCH_NEIGHBOUR                  41    
#define STAT_PORT_DRIVER_INIT_FAIL              42      // FAIL TO INITIALIZE PORT DRIVER
#define STAT_SOFT_KILLED                        43      // FUNCTION INTERRUPTED BY SOFT KILL SIGNAL    
#define STAT_DEPLOY_INTEGRITY_BAD               44
#define STAT_BS_SELECTRULEUNKNOWN               45      // UNKNOWN SELECTION RULE FOR ACTIVE BASE STATION
#define STAT_BS_ROUTESTRATEGYUNKNOWN            46      // UNKNOWN ROUTING STRATEGY 
#define STAT_KEY_MISSING                        47      // AT LEAST ONE KEY IS MISSING
#define STAT_MESSAGE_MISSING                    48      // REQUIRED MESSAGE IS MISSING
#define STAT_NOT_NEIGHBOUR                      49      // NOT COMMUNICATION NEIGBOUR 
#define STAT_INVALID_PTR                        50      // INVALID POINTER DETECTED
#define STAT_KEY_TYPE_BAD                       51      // BAD KEY TYPE

const char* ErrorToString(int error);

#define ERROR_TO_STRING(x) ErrorToString(x)

#endif
