/**
  * @file JVMSimulator.h
  * @author Zdenek Riha
  */

#ifndef JVMSIMULATOR_H
#define JVMSIMULATOR_H

#include <string>
#include <stdint.h>

#include <EACconstants.h>
using namespace std;

// Compile-time parameters
#define MAX_LINE_LENGTH 1024
#define ALLOC_STEP 100
#define EMULATE_MAX_INSTRUCTIONS 1000

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

// Instructions
#define NOP		0x00
#define AALOAD	0x32
#define AASTORE	0x53
#define ALOAD	0x19
#define	ALOAD_0	0x2a
#define	ALOAD_1	0x2b
#define	ALOAD_2	0x2c
#define	ALOAD_3	0x2d
#define	ARETURN	0xb0
#define	ARRAYLENGTH	0xbe
#define	ASTORE	0x3a
#define	ASTORE_0	0x4b
#define	ASTORE_1	0x4c
#define	ASTORE_2	0x4d
#define	ASTORE_3	0x4e
#define	BALOAD	0x33
#define	BASTORE	0x54
#define	BIPUSH	0x10
#define	DUP	0x59
#define	GETSTATIC	0xb2
#define	GOTO	0xa7
#define	I2B	0x91
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
#define	IFEQ		0x99
#define IFNE		0x9a
#define	IF_ICMPGE	0xa2
#define	IF_ICMPLE	0xa4
#define	IF_ICMPNE	0xa0
#define IF_ICMPEQ	0x9f
#define IF_ICMPLT	0xa1
#define IF_ICMPGT	0xa3
#define IINC		0x84
#define ILOAD		0x15
#define	ILOAD_0		0x1a
#define	ILOAD_1		0x1b
#define	ILOAD_2		0x1c
#define	ILOAD_3		0x1d
#define	IMUL		0x68
#define INVOKESPECIAL	0xb7
#define INVOKESTATIC	0xb8
#define IREM			0x70
#define IRETURN			0xac
#define ISHL			0x78
#define ISHR			0x7a
#define IUSHR			0x7c
#define	ISTORE			0x36
#define	ISTORE_0		0x3b
#define	ISTORE_1		0x3c
#define	ISTORE_2		0x3d
#define	ISTORE_3		0x3e
#define ISUB			0x64
#define	IXOR			0x82
#define	MULTIANEWARRAY	0xc5
#define	NEWARRAY		0xbc
#define	PUTSTATIC		0xb3
#define	RETURN			0xb1
#define	SIPUSH			0x11
#define	POP				0x57


// Stack types
#define STACKTYPE_INTEGER	0x1
#define STACKTYPE_ARRAYREF	0x2

// Array types
#define T_INT	10

// Size limitations
#define MAX_NUMBER_OF_LOCAL_VARIABLES 1000

// Structures

//represents one instruction
struct Ins {
    int line_number;
    int instruction_code;
    char *instruction;
    char *param1, *param2;
    char *full_line;
};

//container for instructions
struct I {
    struct Ins **array;
    int filled_elements;
    int maximum_elements;
};

//represents one function with all instructions in list
struct F {
    char *full_name;
    char *short_name;
    struct I *ins_array;
    struct F *next;
};

//stack node
struct element {
    int data_type;
    //unsigned char data;
    int32_t integer;
    struct element *next;
};

//call element stack node
struct call_element {
    char *function;
    int next_line;
    struct call_element *next;
};

//represents current state (which function is used, on which line)
struct Pc {
    char fn[MAX_LINE_LENGTH];
    int ln;
    struct Ins *current_ins;
};

//reprasants global array
struct Ga {
    int32_t *ia;
    signed char *ba;
    int type;
    int number_of_elements;
};


class JVMSimulator {

public:

    //initialize JVMSimulator
    JVMSimulator();

    ~JVMSimulator();

    string shortDescription();

    // initialize JVMSimulator, load dis file, and save all instructions
    int jvmsim_init();

    /**
     * run part of loaded bytecode
     * @param function_number number of function which will be evaluated
     * @param line_from number instruction where evaluation starts
     * @param line_to number of instruction where evaluation ends 
     */
    int jvmsim_run(int function_number, int line_from, int line_to, int use_files);

    //old, won't be used, will be refactored soon
    int jvmsim_main(int argc, char* argv[]);

    /**
     * @return number of functions loaded
     */
    int get_num_of_functions();

    /**
     * @param number_of_function 
     * @return pointer to function
     */
    F* get_function_by_number(unsigned char number_of_function);

    /**
     * Find instruction with instruction number
     * @param fn name of function
     * @param in instruction number
     * @return pointer of instruction
     */
    struct Ins *find_ins(char *fn, int in);

    void call_push(char *fn, int nl);

    int call_pop(struct Pc *PC);

    /**
     * Write all stack values to stdout
     */
    void list_stack();

    /**
     * @return true if stack is empty, false otherwise
     */
    bool stack_empty();

    /**
     * push integer to stack
     * @param value
     */
    void push_int(int32_t value);

    /**
     * push index of referenced array to stack
     * @param value
     */
    void push_arrayref(int32_t value);

    /**
     * pop integer from stack
     * @return value from stack
     */
    int32_t pop_int();

    /**
     * pop index of referenced array
     * @return index
     */
    int32_t pop_arrayref();

    /**
     * emulate specific instruction
     * @param PC current state
     * @return error code - will be refactored soon
     */
    int emulate_ins(struct Pc *PC);

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

    void printl();

    void read_input();

    void write_output();

private:

    //list of all functions
    struct F* m_functions = NULL;

    //number of functions in list
    int	m_numFunctions = 0;

    //stack
    struct element* m_stack = NULL;

    //stack of call elements
    struct call_element* m_call_stack = NULL;

    //array of local variables
    int32_t m_locals[MAX_NUMBER_OF_LOCAL_VARIABLES];

    //meant to be number of local variables stored, but I am not sure if it is correct
    int m_max_local_used = 0;

    //number of stored array
    int m_globalarrays_count = 0;

    //global arrays
    struct Ga m_globalarrays[1000];
};

#endif
