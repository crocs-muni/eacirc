/**
  * @file JVMSimulator.h
  * @author Zdenek Riha
  * @author Michal Hajas
  */

#ifndef JVMSIMULATOR_H
#define JVMSIMULATOR_H

#include <string>
#include <stdint.h>
#include <ctime>

#include <EACconstants.h>
using namespace std;

// Compile-time parameters
#define MAX_LINE_LENGTH 1024
#define ALLOC_STEP 100
#define EMULATE_MAX_INSTRUCTIONS 300

// Return values:
#define ERR_NO_ERROR				0
#define ERR_CANNOT_OPEN_FILE		1
#define ERR_NO_MEMORY				2
#define ERR_DATA_MISMATCH			3
#define ERR_STACK_EMPTY				4
#define ERR_FUNCTION_RETURNED		5
#define ERR_STACK_DATAMISMATCH		6
#define ERR_ARRAYREF_NOT_VALID		7
#define ERR_ARRAYINDEX_OUT_OF_RANGE	8
#define ERR_NOT_IMPLEMENTED			9
#define ERR_NO_SUCH_INSTRUCTION		10
#define ERR_PARSING_INPUT			11
#define ERR_WRITING_OUTPUT			12
#define ERR_MAX_NUMBER_OF_INSTRUCTIONS_EXHAUSTED 13
#define ERR_NO_SUCH_FUNCTION		14
#define ERR_NO_SUCH_LINE            15

//evalution controll
#define CONTINUE            100
#define END                 101
#define GOTO_VALUE          102

// Instructions
#define NOP		0x00
#define LCONST_0    0x09
#define LCONST_1    0x0a
#define AALOAD	0x32
#define AASTORE	0x53
#define ALOAD	0x19
#define	ALOAD_0	0x2a
#define	ALOAD_1	0x2b
#define	ALOAD_2	0x2c
#define	ALOAD_3	0x2d
#define	ARETURN	0xb0
#define	ARRAYLENGTH	0xbe
#define LSTORE  0x37
#define	ASTORE	0x3a
#define LSTORE_0    0x3f
#define LSTORE_1    0x40
#define LSTORE_2    0x41
#define LSTORE_3    0x42
#define	ASTORE_0	0x4b
#define	ASTORE_1	0x4c
#define	ASTORE_2	0x4d
#define	ASTORE_3	0x4e
#define	BALOAD	0x33
#define	CALOAD	0x34
#define	BASTORE	0x54
#define	CASTORE	0x54
#define	BIPUSH	0x10
#define	DUP	0x59
#define DUP2    0x5c
#define	GETSTATIC	0xb2
#define	GOTO	0xa7
#define	I2B	0x91
#define I2C 0x92
#define	IADD	0x60
#define	IAND	0x7e
#define IOR		0x80
#define IALOAD	0x2e
#define	IASTORE	0x4f
#define	ICONST_M1	0x2
#define ICONST_0	0x3
#define	ICONST_1	0x4
#define	ICONST_2	0x5
#define	ICONST_3	0x6
#define	ICONST_4	0x7
#define	ICONST_5	0x8
#define	IDIV		0x6c
#define	LDIV		0x6d
#define LCMP       0x94
#define	IFEQ		0x99
#define IFNE		0x9a
#define IFGE		0x9c
#define	IF_ICMPGE	0xa2
#define	IF_ICMPLE	0xa4
#define	IF_ACMPEQ	0xa5
#define	IF_ICMPNE	0xa0
#define IF_ICMPEQ	0x9f
#define IF_ICMPLT	0xa1
#define IF_ICMPGT	0xa3
#define IINC		0x84
#define LDC         0x12
#define LDC_W       0x13
#define LDC2_W      0x14
#define ILOAD		0x15
#define LLOAD       0x16
#define	ILOAD_0		0x1a
#define	ILOAD_1		0x1b
#define	ILOAD_2		0x1c
#define	ILOAD_3		0x1d
#define LLOAD_0     0x1e
#define LLOAD_1     0x1f
#define LLOAD_2     0x20
#define LLOAD_3     0x21
#define	IMUL		0x68
#define GETFIELD    0xb4
#define PUTFIELD    0xb5
#define INVOKEVIRTUAL   0xb6
#define INVOKESPECIAL	0xb7
#define INVOKESTATIC	0xb8
#define IREM			0x70
#define IRETURN			0xac
#define LREM            0x71
#define ISHL			0x78
#define ISHR			0x7a
#define LSHR			0x7b
#define IUSHR			0x7c
#define LAND            0x7f
#define	ISTORE			0x36
#define	ISTORE_0		0x3b
#define	ISTORE_1		0x3c
#define	ISTORE_2		0x3d
#define	ISTORE_3		0x3e
#define LADD            0x61
#define ISUB			0x64
#define LSUB			0x65
#define	IXOR			0x82
#define I2L             0x85
#define L2I             0x88
#define I2S             0x93
#define IFLE            0x9e
#define	MULTIANEWARRAY	0xc5
#define	NEWARRAY		0xbc
#define	PUTSTATIC		0xb3
#define	RETURN			0xb1
#define NEW             0xbb
#define	SIPUSH			0x11
#define	POP				0x57


// Stack types
#define STACKTYPE_INTEGER	0x1
#define STACKTYPE_ARRAYREF	0x2

// Array types
#define T_INT	10

// Size limitations
#define MAX_NUMBER_OF_VARIABLES  10


// Structures
//represents one instruction
struct Instruction {
    int line_number;
    int instruction_code;
    char* instruction;
    char* param1, * param2;
    char* full_line;
};

//container for instructions
struct Instructions {
    struct Instruction** array;
    int filled_elements;
    int maximum_elements;
};

//represents one function with all instructions in list
struct FunctionNode {
    char* full_name;
    char* short_name;
    struct Instructions* ins_array;
    struct FunctionNode* next;
};

//stack node
struct StackNode {
    int data_type;
    //unsigned char data;
    int32_t integer;
    struct StackNode* next;
};

//call element stack node
struct CallStackNode {
    struct FunctionNode* function;
    int next_line;
    struct CallStackNode* next;
};

//represents current state (which function is used, on which line)
struct Pc {
    struct FunctionNode* currentFunction;
    int ln;
};

//represents global array
struct GlobalArray {
    int32_t* int_array;
    signed char* ba;
    int type;
    int number_of_elements;
};

class JVMSimulator {

public:

    //initialize JVMSimulator
    JVMSimulator();

    ~JVMSimulator();

    /**
     * @brief jvmsim_init parse fileName file
     * @param fileName to parse bytecode from
     * @return error code
     */
    int jvmsim_init(string fileName);

    /**
    * run part of loaded bytecode
    * @param function_number number of function which will be evaluated
    * @param line_from number instruction where evaluation starts
    * @param line_to number of instruction where evaluation ends
    */
    int jvmsim_run(int function_number, int line_from, int line_to);

    /**
    * @return number of functions loaded
    */
    int get_num_of_functions();

    /**
    * @param number_of_function
    * @return pointer to function
    */
    struct FunctionNode* get_function_by_number(unsigned char number_of_function);

    /**
    * @param name
    * @return pointer to function
    */
    struct FunctionNode* get_function_by_name(char* name);

    /**
    * @param function within which is find instruction
    * @param instructionNumber is number assigned by compilator
    * @return line number within function
    */
    int find_instruction_by_number(FunctionNode* function, int instructionNumber);

    /**
    * Saves current state to call stack
    * @param PC current state
    */
    void call_push(struct Pc* PC);

    /**
    * Loads state from call stack and save it to PC param
    */
    int call_pop(struct Pc* PC);

    /**
    * Write all stack values to stdout
    */
    void list_stack();

    /**
    * @return true if stack is empty, false otherwise
    */
    bool stack_empty();

    /**
    * @return true if call stack is empty, false otherwise
    */
    bool callStackEmpty();

    /**
    * push integer to stack
    * @param value
    */
    void push_int(int32_t value);

    /**
    * push long to stack
    * @param value
    */
    void push_long(int64_t value);

    /**
    * push index of referenced array to stack
    * @param value
    */
    void push_arrayref(int32_t value);

    /**
    * pop integer from stack
    * @return value from stack
    */
    void pop_int(int32_t& returnValue);

    /**
    * pop long from stack
    * @return value from stack
    */
    void pop_long(int64_t& returnValue);

    /**
    * pop index of referenced array
    * @return index
    */
    void pop_arrayref(int32_t& returnValue);

    /**
    * emulate specific instruction
    * @param PC current state
    * @return error code - will be refactored soon
    */
    int emulate_ins(struct Pc* PC, int& returnValue);

    /**
    * @param c
    * @return 1 if c is white place, 0 otherwise
    */
    int white(char c);

    /**
    * @param i
    * @return indentificator of specific instruction
    */
    int code(char* i);

    /**
    * Find out if top of stack is of type array reference
    */
    bool hasArrayRefOnStack();

    /**
    * Find out if there are four integers on top of stack
    */
    bool hasFourIntegersOnStack();

    bool hasThreeIntegersOnStack();

    /**
    * Find out if there are two integers on top of stack
    */
    bool hasTwoIntegersOnStack();

    /**
    * Find out if there is integer on top of stack
    */
    bool hasIntegerOnStack();

    bool hasTwoArrRefOnStack();

private:

    /**
    * @return index of first non-used element of globalArray
    */
    int get_next_global_index();

    /**
    * Remove state from last run global arrays and local variables
    */
    void destroy_state();

    //list of all functions
    struct FunctionNode* m_functions = NULL;

    //number of functions in list
    int m_numFunctions = 0;

    //stack
    struct StackNode* m_stack = NULL;

    //stack of call elements
    struct CallStackNode* m_callStack = NULL;

    //array of local variables
    int32_t m_locals[MAX_NUMBER_OF_VARIABLES];

    //meant to be number of local variables stored, but I am not sure if it is correct
    bool m_localsUsed[MAX_NUMBER_OF_VARIABLES];

    long createLongFromTwoInts(int32_t first, int32_t second);
    //global arrays
    struct GlobalArray m_globalArrays[MAX_NUMBER_OF_VARIABLES];

    bool m_globalArraysUsed[MAX_NUMBER_OF_VARIABLES];
};

#endif
