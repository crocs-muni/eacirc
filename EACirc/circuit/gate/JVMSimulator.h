/**
  * @file JVMSimulator.h
  * @author Petr Svenda
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
struct Ins {
	int line_number;
	int instruction_code;
	char *instruction;
	char *param1, *param2;
	char *full_line;
};

struct I {
	struct Ins **array;
	int filled_elements;
	int maximum_elements;
};

struct F {
	char *full_name;
	char *short_name;
	struct I *ins_array;
	struct F *next;
};

struct element {
	int data_type;
	//unsigned char data;
	int32_t integer;
	struct element *next;
};

struct call_element {
	char *function;
	int next_line;
	struct call_element *next;
};


struct Pc {
	char fn[MAX_LINE_LENGTH];
	int ln;
	struct Ins *current_ins;
};

struct Ga {
	int32_t *ia;
	signed char *ba;
	int type;
	int number_of_elements;
};


class JVMSimulator {
	struct F *Functions = NULL;
	int	numFunctions = 0;
	struct element *Stack = NULL;
	struct call_element *Call_stack = NULL;

	int32_t locals[MAX_NUMBER_OF_LOCAL_VARIABLES];
	int max_local_used = 0;
	int globalarrays_count = 0;
	struct Ga globalarrays[1000];

public:
    JVMSimulator();
    ~JVMSimulator();
    string shortDescription();

	int jvmsim_init();
	int jvmsim_run(string function_name, int line_from, int line_to, int use_files);
	int jvmsim_main(int argc, char* argv[]);

	string getFunctionNameByID(int functionID);

	inline struct Ins *find_ins(char *fn, int in);
	inline void call_push(char *fn, int nl);
	inline int call_pop(struct Pc *PC);
	inline void list_stack();
	bool stack_empty();
	inline void push_int(int ii);
	inline void push_arrayref(int32_t ii);
	inline int32_t pop_int();
	inline int32_t pop_arrayref();
	inline void emulate_ins(struct Pc *PC);
	

	int white(char c);
	int code(char* i);
	void printl();
	void read_input();
	void write_output();
};




#endif