#include "JVMSimulator.h"
#include "EACglobals.h"
#include <cassert>

#pragma warning(disable:4996)

JVMSimulator::JVMSimulator() {
	int retval = jvmsim_init();
	if (retval == 0) {
		mainLogger.out(LOGGER_INFO) << "JVM simulator initialized." << endl;
	}
	else {
		mainLogger.out(LOGGER_ERROR) << "JVM simulator initialisation returned error (" << retval << ")" << endl;
	}
}

JVMSimulator::~JVMSimulator() {
}

// Utilities
struct Ins *JVMSimulator::find_ins(char *fn, int instructionLineNumber)
{
	struct F *fnct= Functions;
	int found=0;
	while(fnct)
	{
		if(!strcmp(fnct->short_name,fn))
		{
			found=1;
			break;
		}
		fnct=fnct->next;
	}
	if(!found) return NULL;


	for (int a = 0; a < fnct->ins_array->filled_elements; a++) {
		if (fnct->ins_array->array[a]->line_number == instructionLineNumber) {
			return fnct->ins_array->array[a];
		}
	}
	return NULL;
}

void JVMSimulator::call_push(char *fn, int nl)
{
	struct call_element *n=(struct call_element*)malloc(sizeof(struct call_element));
	if (n == NULL) {
		mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
		exit(ERR_NO_MEMORY);
	}
	n->next=Call_stack;
	n->function=(char*)malloc(strlen(fn)+1);
	if (n->function == NULL) {
		mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
		exit(ERR_NO_MEMORY);
	}
	strcpy(n->function,fn);
	n->next_line=nl;
	Call_stack=n;
}

int JVMSimulator::call_pop(struct Pc *PC)
{
	if(Call_stack==NULL) return 1;
	strcpy(PC->fn,Call_stack->function);
	PC->ln=Call_stack->next_line;
	PC->current_ins=find_ins(PC->fn,PC->ln);
	if (PC->current_ins == NULL) {
		mainLogger.out(LOGGER_WARNING) << "Cannot find instruction during POP operation. Returning 1 (call stack empty)." << endl;
		//exit(ERR_DATA_MISMATCH);
		return 1;
	}
	struct call_element *d=Call_stack;
	Call_stack=Call_stack->next;
	delete d;
	return 0;
}


void JVMSimulator::list_stack()
{
	printf("S: ");
	struct element *n=Stack;
	while(n)
	{
		printf("-> %i",(int)n->integer);
		n=n->next;
	}
	printf("\n");
}

bool JVMSimulator::stack_empty()
{
	return Stack == NULL;
}

void JVMSimulator::push_int(int32_t ii)
{
	struct element *n=(struct element*)malloc(sizeof(struct element));
	if (n == NULL) {
		mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
		exit(ERR_NO_MEMORY);
	}
	n->next=Stack;
	n->integer=ii;
	n->data_type=STACKTYPE_INTEGER;
	Stack=n;
}

void JVMSimulator::push_arrayref(int32_t ii)
{
	struct element *n=(struct element*)malloc(sizeof(struct element));
	if (n == NULL) {
		mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
		exit(ERR_NO_MEMORY);
	}
	n->next=Stack;
	n->integer=ii;
	n->data_type=STACKTYPE_ARRAYREF;
	Stack=n;
}

/*
void push(unsigned char c)
{
	struct element *n=(struct element*)malloc(sizeof(struct element));
	if(n==NULL) 
	{
		mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
		exit(ERR_NO_MEMORY);
	}
	n->next=Stack;
	n->data=c;
	Stack=n;
}


void push_byte(signed char c)
{
	push((unsigned char)c);
}
*/

int32_t JVMSimulator::pop_int()
{
	if (Stack == NULL)
	{
		//exit(ERR_STACK_EMPTY);
		mainLogger.out(LOGGER_WARNING) << "Stack empty during POP. Returning 0." << endl;
		return 0;
	}
		
	if(Stack->data_type!=STACKTYPE_INTEGER) {
		//exit(ERR_STACK_DATAMISMATCH);
		mainLogger.out(LOGGER_WARNING) << "Data mismatch during POP. Returning 0." << endl;
		return 0;
	}
	int32_t res = Stack->integer;
	struct element *d = Stack;
	Stack=Stack->next;
	delete d;
	return res;
}

int32_t JVMSimulator::pop_arrayref()
{
	if (Stack == NULL) {
		//exit(ERR_STACK_EMPTY);
		mainLogger.out(LOGGER_WARNING) << "Stack empty during POP ARRAYREF. Returning 0." << endl;
		return 0;
	}
	if (Stack->data_type != STACKTYPE_ARRAYREF) {
		//exit(ERR_STACK_DATAMISMATCH);
		mainLogger.out(LOGGER_WARNING) << "Data mismatch during POP ARRAYREF. Returning 0." << endl;
		return 0;
	}
	int32_t res = Stack->integer;
	struct element *d = Stack;
	Stack=Stack->next;
	delete d;
	return res;
}

/*
unsigned char pop()
{
	if(Stack==NULL) exit(ERR_STACK_EMPTY);
	unsigned char res = Stack->data;
	struct element *d = Stack;
	Stack=Stack->next;
	delete d;
	return res;
}

signed char pop_byte()
{
	return (signed char) pop();
}
*/

/*
int32_t pop_int()
{
	unsigned char a1=pop();
	unsigned char a2=pop();
	unsigned char a3=pop();
	unsigned char a4=pop();
	int32_t res=(a1<<24)+(a2<<16)+(a3<<8)+a4;
	return res;
}


void push_int(int32_t i)
{
	unsigned char a1=i>>24;
	unsigned char a2=(i>>16)&&0xFF;
	unsigned char a3=(i>>8)&&0xFF;
	unsigned char a4=i&0xFF;
	push(a4); push(a3); push(a2); push(a1);
}
*/

/*
void fill_stack()
{
	for (int i=0;i<100;i++)
		push(rand()%256);
}
*/

int JVMSimulator::emulate_ins(struct Pc *PC)
{
	//list_stack();
	//printl();
	//printf("Emulating: %s\n",PC->current_ins->full_line);
	int jump = 0;
	switch (PC->current_ins->instruction_code)
	{
	case NOP: {
				  break;
	}
	case IADD:
	{
				 int32_t a = pop_int();
				 int32_t b = pop_int();
				 int32_t c = a + b;
				 push_int(c);
	}
		break;
	case ISUB:
	{
				 int32_t a = pop_int();
				 int32_t b = pop_int();
				 int32_t c = b - a;
				 push_int(c);
	}
		break;
	case IAND:
	{
				 int32_t a = pop_int();
				 int32_t b = pop_int();
				 int32_t c = a&b;
				 push_int(c);
	}
		break;
	case IXOR:
	{
				 int32_t a = pop_int();
				 int32_t b = pop_int();
				 int32_t c = a^b;
				 push_int(c);
	}
		break;
	case ICONST_M1:
		push_int((int32_t)-1);
		break;
	case ICONST_0:
		push_int((int32_t)0);
		break;
	case ICONST_1:
		push_int((int32_t)1);
		break;
	case ICONST_2:
		push_int((int32_t)2);
		break;
	case ICONST_3:
		push_int((int32_t)3);
		break;
	case ICONST_4:
		push_int((int32_t)4);
		break;
	case ICONST_5:
		push_int((int32_t)5);
		break;
	case I2B:
	{
				int32_t a = pop_int();
				signed char c = (signed char)a;
				push_int((int)c);
	}
		break;
	case IDIV:
	{
				 int32_t a = pop_int();
				 int32_t b = pop_int();
				 int32_t c = b / a;
				 push_int(c);
	}
		break;
	case IMUL:
	{
				 int32_t a = pop_int();
				 int32_t b = pop_int();
				 int32_t c = b*a;
				 push_int(c);
	}
		break;
	case IREM:
	{
				 int32_t a = pop_int();
				 int32_t b = pop_int();
				 int32_t c = b%a;
				 push_int(c);
	}
		break;
	case ISHL:
	{
				 int32_t a = pop_int();
				 int32_t b = pop_int();
				 int32_t c = b << (a & 0x1f);
				 push_int(c);
	}
		break;
	case ISHR:
	{
				 int32_t a = pop_int();
				 int32_t b = pop_int();
				 int32_t c = b >> (a & 0x1f);
				 push_int(c);
	}
		break;
	case BIPUSH:
	{
				   int j = atoi(PC->current_ins->param1);
				   signed char c = (signed char)j;
				   push_int((int32_t)c);
	}
		break;
	case SIPUSH:
	{
				   int j = atoi(PC->current_ins->param1);
				   int16_t s = (int16_t)j;
				   push_int((int32_t)s);
	}
		break;
	case DUP:
	{
				int j = pop_int();
				push_int(j);
				push_int(j);
	}
		break;
	case GOTO:
	{
				 PC->ln = atoi(PC->current_ins->param1);
				 PC->current_ins = find_ins(PC->fn, PC->ln);
				 if (PC->current_ins == NULL) {
					 mainLogger.out(LOGGER_WARNING) << "Wrong GOTO. Interrupting execution." << endl;
					 //exit(ERR_DATA_MISMATCH);
					 return -1;
				 }
				 jump = 1;
	}
		break;
	case IFEQ:
	{
				 int32_t value = pop_int();
				 if (value == 0)
				 {
					 PC->ln = atoi(PC->current_ins->param1);
					 PC->current_ins = find_ins(PC->fn, PC->ln);
					 if (PC->current_ins == NULL){
						 mainLogger.out(LOGGER_WARNING) << "Wrong IFEQ jump. Interrupting execution." << endl;
						 //exit(ERR_DATA_MISMATCH);
						 return -1;
					 }
					 jump = 1;
				 }
	}
		break;
	case IF_ICMPGE:
	{
					  int32_t value2 = pop_int();
					  int32_t value1 = pop_int();
					  if (value1 >= value2)
					  {
						  PC->ln = atoi(PC->current_ins->param1);
						  PC->current_ins = find_ins(PC->fn, PC->ln);
						  if (PC->current_ins == NULL){
							  mainLogger.out(LOGGER_WARNING) << "Wrong IF_ICMPGE jump. Interrupting execution." << endl;
							  //exit(ERR_DATA_MISMATCH);
							  return -1;
						  }
						  jump = 1;
					  }
	}
		break;
	case INVOKESTATIC:
	{
		{
			struct F *fnct = Functions;
			int found = 0;
			while (fnct)
			{
				if (!strcmp(fnct->short_name, PC->fn))
				{
					found = 1;
					break;
				}
				fnct = fnct->next;
			}
			if (!found) {
				mainLogger.out(LOGGER_WARNING) << "Wrong INVOKESTATIC jump. Interrupting execution." << endl;
				//exit(ERR_DATA_MISMATCH);
				return -1;
			}
			int step_done = 0;
			for (int a = 0; a<fnct->ins_array->filled_elements; a++)
			if (fnct->ins_array->array[a]->line_number == PC->ln)
			{
				call_push(PC->fn, fnct->ins_array->array[a + 1]->line_number);
				step_done = 1;
				break;
			}
			if (!step_done) {
				mainLogger.out(LOGGER_WARNING) << "Data mismatch during INVOKESTATIC jump. Interrupting execution." << endl;
				//exit(ERR_DATA_MISMATCH);
				return -1;
			}
		}
						 char cp[MAX_LINE_LENGTH], *p1, *p2;
						 strcpy(cp, PC->current_ins->full_line);
						 p1 = strstr(cp, "//Method ");
						 if (!p1) {
							 mainLogger.out(LOGGER_WARNING) << "Data mismatch during INVOKESTATIC jump. Interrupting execution." << endl;
							 //exit(ERR_DATA_MISMATCH);
							 return -1;
						 }
						 p1 += 9;
						 p2 = strchr(p1, ':');
						 if (!p2) {
							 mainLogger.out(LOGGER_WARNING) << "Data mismatch during INVOKESTATIC jump. Interrupting execution." << endl;
							 //exit(ERR_DATA_MISMATCH);
							 return -1;
						 }
						 *p2 = 0;
						 strcpy(PC->fn, p1); PC->ln = 0;
						 PC->current_ins = find_ins(PC->fn, PC->ln);
						 if (PC->current_ins == NULL) {
							 mainLogger.out(LOGGER_WARNING) << "Data mismatch during INVOKESTATIC jump. Interrupting execution." << endl;
							 //exit(ERR_DATA_MISMATCH);
							 return -1;
						 }
						 jump = 1;
	}
		break;
	case RETURN:
	case ARETURN:
	{
					if (call_pop(PC) == 1){
						//exit(ERR_FUNCTION_RETURNED);
						return 1;
					}
					jump = 1;
	}
		break;
	case IRETURN:
	{
					if (call_pop(PC) == 1)
					{
						// int32_t r=pop_int();
						// printf("Returning result: %i\n",(int)r);
						write_output();
						//exit(ERR_FUNCTION_RETURNED);
						return 1;
					}

					jump = 1;
	}
		break;
	case ILOAD_0:
	{
					push_int(locals[0]);
					if (max_local_used<0)max_local_used = 0;
	}
		break;
	case ILOAD_1:
	{
					push_int(locals[1]);
					if (max_local_used<1)max_local_used = 1;
	}
		break;
	case ILOAD_2:
	{
					push_int(locals[2]);
					if (max_local_used<2)max_local_used = 2;
	}
		break;
	case ILOAD_3:
	{
					push_int(locals[3]);
					if (max_local_used<3)max_local_used = 3;
	}
		break;
	case ILOAD:
	{
				  // todo: verify index...
				  push_int((int32_t)locals[atoi(PC->current_ins->param1)]);
				  if (max_local_used<atoi(PC->current_ins->param1))max_local_used = atoi(PC->current_ins->param1);
	}
		break;
	case ISTORE:
	{
				   int32_t i = pop_int();
				   locals[atoi(PC->current_ins->param1)] = i;
				   if (max_local_used<atoi(PC->current_ins->param1))max_local_used = atoi(PC->current_ins->param1);
	}
		break;
	case ISTORE_0:
	{
					 int32_t i = pop_int();
					 locals[0] = i;
					 if (max_local_used<0)max_local_used = 0;
	}
		break;
	case ISTORE_1:
	{
					 int32_t i = pop_int();
					 locals[1] = i;
					 if (max_local_used<1)max_local_used = 1;
	}
		break;
	case ISTORE_2:
	{
					 int32_t i = pop_int();
					 locals[2] = i;
					 if (max_local_used<2)max_local_used = 2;
	}
		break;
	case ISTORE_3:
	{
					 int32_t i = pop_int();
					 locals[3] = i;
					 if (max_local_used<3)max_local_used = 3;
	}
		break;
	case NEWARRAY:
	{
					 int count = pop_int();
					 if (!strcmp(PC->current_ins->param1, "integer"))
					 {
						 globalarrays[globalarrays_count].ia = (int32_t*)malloc(count*sizeof(int32_t));
						 if (globalarrays[globalarrays_count].ia == NULL) {
							 mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
							 exit(ERR_NO_MEMORY);
						 }
						 globalarrays[globalarrays_count].type = T_INT;
						 globalarrays[globalarrays_count].number_of_elements = count;
						 push_arrayref(globalarrays_count);
						 globalarrays_count++;
					 }
	}
		break;
	case ASTORE_0:
	{
					 int32_t ref = pop_arrayref();
					 locals[0] = ref;
					 if (max_local_used<0)max_local_used = 0;
	}
		break;
	case ASTORE_1:
	{
					 int32_t ref = pop_arrayref();
					 locals[1] = ref;
					 if (max_local_used<1)max_local_used = 1;
	}
		break;
	case ASTORE_2:
	{
					 int32_t ref = pop_arrayref();
					 locals[2] = ref;
					 if (max_local_used<2)max_local_used = 2;
	}
		break;
	case ASTORE_3:
	{
					 int32_t ref = pop_arrayref();
					 locals[3] = ref;
					 if (max_local_used<3)max_local_used = 3;
	}
		break;
	case ASTORE:
	{
				   int32_t ref = pop_arrayref();
				   // verify index
				   locals[atoi(PC->current_ins->param1)] = ref;
				   if (max_local_used<atoi(PC->current_ins->param1))max_local_used = atoi(PC->current_ins->param1);
	}
		break;
	case AALOAD:
	{
				   int index = pop_int();
				   int32_t ref = pop_arrayref();
				   if (ref >= globalarrays_count) {
					   mainLogger.out(LOGGER_WARNING) << "Array reference is not valid in AALOAD. Interrupting execution." << endl;
					   //exit(ERR_ARRAYREF_NOT_VALID);
					   return -1;
				   }
				   if (index >= globalarrays[ref].number_of_elements) {
					   mainLogger.out(LOGGER_WARNING) << "Array index is out of range in AALOAD. Interrupting execution." << endl;
					   //exit(ERR_ARRAYINDEX_OUT_OF_RANGE);
					   return -1;
				   }
				   if (globalarrays[ref].type == T_INT)
				   {
					   push_int(globalarrays[ref].ia[index]);

				   }
				   else 
				   { 
					   /*TODO: handle also other data types */ 
					   mainLogger.out(LOGGER_WARNING) << "This data type is not implemented for AALOAD. Interrupting execution." << endl;
					   //exit(ERR_NOT_IMPLEMENTED); 
					   return 1;
				   }
	}
		break;
	default:
		break;
	}

	// prepare pointer to the next instruction
	if (!jump)
	{
		struct F *fnct = Functions;
		int found = 0;
		while (fnct)
		{
			if (!strcmp(fnct->short_name, PC->fn))
			{
				found = 1;
				break;
			}
			fnct = fnct->next;
		}
		if (!found) {
			mainLogger.out(LOGGER_WARNING) << "Data mismatch in parsing the source code. Interrupting execution." << endl;
			//exit(ERR_DATA_MISMATCH);
			return -1;
		}
		int step_done = 0;
		for (int a = 0; a<fnct->ins_array->filled_elements; a++)
		if (fnct->ins_array->array[a]->line_number == PC->ln)
		{
			PC->ln = fnct->ins_array->array[a + 1]->line_number;
			PC->current_ins = fnct->ins_array->array[a + 1];
			step_done = 1;
			break;
		}
		if (!step_done) {
			mainLogger.out(LOGGER_WARNING) << "Data mismatch in parsing the source code. Interrupting execution." << endl;
			//exit(ERR_DATA_MISMATCH);
			return -1;
		}
	}
	return 0;
}


int JVMSimulator::white(char c)
{
	if(c==' '||c=='\t'||c=='\r'||c=='\n') return 1;
	return 0;
}

int JVMSimulator::code(char* i)
{
	if(!strcmp(i,"aaload"))return AALOAD;
	if(!strcmp(i,"aastore"))return AASTORE;
	if(!strcmp(i,"aload"))return ALOAD;
	if(!strcmp(i,"aload_0"))return ALOAD_0;
	if(!strcmp(i,"aload_1"))return ALOAD_1;
	if(!strcmp(i,"aload_2"))return ALOAD_2;
	if(!strcmp(i,"aload_3"))return ALOAD_3;
	if(!strcmp(i,"areturn"))return ARETURN;
	if(!strcmp(i,"arraylength"))return ARRAYLENGTH;
	if(!strcmp(i,"astore"))return ASTORE;
	if(!strcmp(i,"astore_0"))return ASTORE_0;
	if(!strcmp(i,"astore_1"))return ASTORE_1;
	if(!strcmp(i,"astore_2"))return ASTORE_2;
	if(!strcmp(i,"astore_3"))return ASTORE_3;
	if(!strcmp(i,"baload"))return BALOAD;
	if(!strcmp(i,"bastore"))return BASTORE;
	if(!strcmp(i,"bipush"))return BIPUSH;
	if(!strcmp(i,"dup"))return DUP;
	if(!strcmp(i,"getstatic"))return GETSTATIC;
	if(!strcmp(i,"goto"))return GOTO;
	if(!strcmp(i,"i2b"))return I2B;
	if(!strcmp(i,"iadd"))return IADD;
	if(!strcmp(i,"iand"))return IAND;
	if(!strcmp(i,"iaload"))return IALOAD;
	if(!strcmp(i,"iastore"))return IASTORE;
	if(!strcmp(i,"iconst_m1"))return ICONST_M1;
	if(!strcmp(i,"iconst_0"))return ICONST_0;
	if(!strcmp(i,"iconst_1"))return ICONST_1;
	if(!strcmp(i,"iconst_2"))return ICONST_2;
	if(!strcmp(i,"iconst_3"))return ICONST_3;
	if(!strcmp(i,"iconst_4"))return ICONST_4;
	if(!strcmp(i,"iconst_5"))return ICONST_5;
	if(!strcmp(i,"idiv"))return IDIV;
	if(!strcmp(i,"ifeq"))return IFEQ;
	if(!strcmp(i,"ifne"))return IFNE;
	if(!strcmp(i,"if_icmpge"))return IF_ICMPGE;
	if(!strcmp(i,"if_icmple"))return IF_ICMPLE;
	if(!strcmp(i,"if_icmpne"))return IF_ICMPNE;
	if(!strcmp(i,"iinc"))return IINC;
	if(!strcmp(i,"iload"))return ILOAD;
	if(!strcmp(i,"iload_0"))return ILOAD_0;
	if(!strcmp(i,"iload_1"))return ILOAD_1;
	if(!strcmp(i,"iload_2"))return ILOAD_2;
	if(!strcmp(i,"iload_3"))return ILOAD_3;
	if(!strcmp(i,"imul"))return IMUL;
	if(!strcmp(i,"invokespecial"))return INVOKESPECIAL;
	if(!strcmp(i,"invokestatic"))return INVOKESTATIC;
	if(!strcmp(i,"irem"))return IREM;
	if(!strcmp(i,"ireturn"))return IRETURN;
	if(!strcmp(i,"ishl"))return ISHL;
	if(!strcmp(i,"ishr"))return ISHR;
	if(!strcmp(i,"istore"))return ISTORE;
	if(!strcmp(i,"istore_0"))return ISTORE_0;
	if(!strcmp(i,"istore_1"))return ISTORE_1;
	if(!strcmp(i,"istore_2"))return ISTORE_2;
	if(!strcmp(i,"istore_3"))return ISTORE_3;
	if(!strcmp(i,"isub"))return ISUB;
	if(!strcmp(i,"ixor"))return IXOR;
	if(!strcmp(i,"multianewarray"))return MULTIANEWARRAY;
	if(!strcmp(i,"newarray"))return NEWARRAY;
	if(!strcmp(i,"pop"))return POP;
	if(!strcmp(i,"putstatic"))return PUTSTATIC;
	if(!strcmp(i,"return"))return RETURN;
	if(!strcmp(i,"sipush"))return SIPUSH;

	assert(false); 
	exit(ERR_DATA_MISMATCH);
}
/*
void printl()
{
printf("Locals - aa: %i, bb: %i, r: %i, t: %i\n", locals[2], locals[3], locals[4], locals[5]);
}
*/




void JVMSimulator::read_input()
{
	FILE *f=fopen("input.txt","rt");
	if (f == NULL){
		mainLogger.out(LOGGER_ERROR) << "Error parsing the imput file. Cannot open the file. Exitting..." << endl;
		exit(ERR_PARSING_INPUT);
	}
	char number[MAX_LINE_LENGTH],*p;
	
	int to_continue=1;
	int local_index=0;
	do{
        strcpy(number,"");
		p=number;
		int next_word=0;
		do
		{
			int c=fgetc(f);
			if (c == EOF){
				mainLogger.out(LOGGER_ERROR) << "Error parsing the imput file. Exitting..." << endl;
				exit(ERR_PARSING_INPUT);
			}
			if(c=='\n'||c=='\r')to_continue=0;
			if(c==' ')next_word=1;
			*p=c; p++; *p=0;
		}while(!next_word&&to_continue);
		if (to_continue)
		{
			locals[local_index] = (int32_t)strtol(number, NULL, 16);
			local_index++;
		}
	}while(to_continue);

	to_continue=1;
	do
	{
        strcpy(number,"");
		p=number;
		int next_word=0;
		do
		{
			int c=fgetc(f);
			if(c==EOF||c=='\n'||c=='\r')to_continue=0;
			if(c==' ')next_word=1;
			*p=c; p++; *p=0;
        }while(!next_word&&to_continue);
		if (to_continue)
		{
			push_int((int32_t)strtol(number, NULL, 16));
		}
	}while(to_continue);
	
	//list_stack();
	fclose(f);
}

void JVMSimulator::write_output()
{
	FILE *f=fopen("output.txt","wt");
	if (f == NULL){
		mainLogger.out(LOGGER_ERROR) << "Error writing the output file. Exitting..." << endl;
		exit(ERR_WRITING_OUTPUT);
	}

	struct element *n=Stack;
	while(n)
	{
		switch(n->data_type)
		{
		case STACKTYPE_INTEGER:
			fprintf(f,"%x ",(int)n->integer);
			break;
		case STACKTYPE_ARRAYREF:
			fprintf(f,"%x(ARRAYREF) ",(int)n->integer);
			break;
		}
		n=n->next;
	}
	fprintf(f,"\n");
	for(int i=0;i<=max_local_used;i++)
		fprintf(f,"%x ", locals[i]);
	fclose(f);
}

int JVMSimulator::jvmsim_init()
{
	FILE *f = fopen("AES.dis", "rt");
	if (f == NULL) return(ERR_CANNOT_OPEN_FILE);
	char line[MAX_LINE_LENGTH], lastline[MAX_LINE_LENGTH];

	// read lines and parse
	strcpy(line, "");
	while (strcpy(lastline, line), fgets(line, MAX_LINE_LENGTH, f) != NULL)
	{
		// remove new line characters
		size_t len = strlen(line);
		if (len >= 1 && (line[len - 1] == '\n' || line[len - 1] == '\r'))line[len - 1] = 0;
		if (len >= 2 && (line[len - 2] == '\n' || line[len - 1] == '\r'))line[len - 2] = 0;

		// parse the line
		if (!strcmp(line, ""))continue;
		if (!strcmp(line, "}"))continue;
		if (!strncmp(line, "Compiled from", strlen("Compiled from")))continue;
		if (!strncmp(line, "public class ", strlen("public class ")))continue;
		if (!strncmp(line, "public", strlen("public")))continue;
		if (!strncmp(line, "static", strlen("static")))continue;
		if (!strncmp(line, "  Code:", strlen("Code:")))
		{
			// new function
			struct F *fnew = (struct F*)malloc(sizeof(struct F));
			if (fnew == NULL)
			{
				mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
				return(ERR_NO_MEMORY);
			}
			fnew->next = Functions;
			Functions = fnew;
			fnew->full_name = (char*)malloc(strlen(lastline) + 1);
			if (fnew->full_name == NULL)
			{
				mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
				return(ERR_NO_MEMORY);
			}
			strcpy(fnew->full_name, lastline);
			fnew->short_name = (char*)malloc(strlen(lastline) + 1);
			if (fnew->short_name == NULL) return(ERR_NO_MEMORY);
			strcpy(fnew->short_name, "");
			char copyline[MAX_LINE_LENGTH], *p1, *p2;
			strcpy(copyline, lastline);
			p1 = strchr(copyline, '(');
			if (p1)
			{
				p2 = p1;
				*p1 = 0;
				while (p2 != copyline && *p2 != ' ')p2--;
				if (*p2 == ' ') strcpy(fnew->short_name, p2 + 1);
			}
			fnew->ins_array = (struct I*)malloc(sizeof(struct I));
			if (fnew->ins_array == NULL)
			{
				mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
				return(ERR_NO_MEMORY);
			}
			fnew->ins_array->array = (struct Ins**)malloc(ALLOC_STEP*sizeof(struct Ins*));
			fnew->ins_array->filled_elements = 0;
			fnew->ins_array->maximum_elements = ALLOC_STEP;

			numFunctions++;
			continue;
		}
		// instructions
		if (Functions == NULL)
			return(ERR_DATA_MISMATCH);
		if (Functions->ins_array->filled_elements >= Functions->ins_array->maximum_elements)
		{
			Functions->ins_array->array = (struct Ins**)realloc(Functions->ins_array->array, (Functions->ins_array->maximum_elements + ALLOC_STEP)*sizeof(struct Ins*));
			if (Functions->ins_array->array == NULL)
			{
				mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
				return(ERR_NO_MEMORY);
			}
			Functions->ins_array->maximum_elements += ALLOC_STEP;
		}
		Functions->ins_array->array[Functions->ins_array->filled_elements] = (struct Ins*)malloc(sizeof(struct Ins));
		if (Functions->ins_array->array[Functions->ins_array->filled_elements] == NULL)
		{
			mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
			return(ERR_NO_MEMORY);
		}
		Functions->ins_array->array[Functions->ins_array->filled_elements]->full_line = (char*)malloc(strlen(line) + 1);
		if (Functions->ins_array->array[Functions->ins_array->filled_elements]->full_line == NULL)
		{
			mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
			return(ERR_NO_MEMORY);
		}
		strcpy(Functions->ins_array->array[Functions->ins_array->filled_elements]->full_line, line);
		Functions->ins_array->array[Functions->ins_array->filled_elements]->line_number = atoi(line);
		Functions->ins_array->array[Functions->ins_array->filled_elements]->instruction = (char*)malloc(strlen(line) + 1);
		if (Functions->ins_array->array[Functions->ins_array->filled_elements]->instruction == NULL)
		{
			mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
			return(ERR_NO_MEMORY);
		}
		char copyline[MAX_LINE_LENGTH], *p1, *p2;
		strcpy(copyline, line);
		p1 = strchr(copyline, ':');
		if (!p1)
			return(ERR_DATA_MISMATCH);
		p1++;
		while (white(*p1))p1++;
		p2 = p1;
		while (*p2&&!white(*p2))p2++;
		char s = *p2; *p2 = 0;
		strcpy(Functions->ins_array->array[Functions->ins_array->filled_elements]->instruction, p1);
		Functions->ins_array->array[Functions->ins_array->filled_elements]->instruction_code = code(Functions->ins_array->array[Functions->ins_array->filled_elements]->instruction);
		Functions->ins_array->array[Functions->ins_array->filled_elements]->param1 = (char*)malloc(strlen(line) + 1);
		if (Functions->ins_array->array[Functions->ins_array->filled_elements]->param1 == NULL)
		{
			mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
			return(ERR_NO_MEMORY);
		}
		strcpy(Functions->ins_array->array[Functions->ins_array->filled_elements]->param1, "");
		Functions->ins_array->array[Functions->ins_array->filled_elements]->param2 = (char*)malloc(strlen(line) + 1);
		if (Functions->ins_array->array[Functions->ins_array->filled_elements]->param2 == NULL)
		{
			mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
			return(ERR_NO_MEMORY);
		}
		strcpy(Functions->ins_array->array[Functions->ins_array->filled_elements]->param2, "");
		*p2 = s;
		if (*p2)
		{
			while (*p2&&white(*p2))p2++;
			if (*p2)
			{
				char *p3 = p2;
				while (*p3 && (!white(*p3) && *p3 != ','&&*p3 != ';'))p3++;
				s = *p3; *p3 = 0;
				strcpy(Functions->ins_array->array[Functions->ins_array->filled_elements]->param1, p2);
				*p3 = s;
				if (*p3 == ',')
				{
					p3++;
					while (*p3&&white(*p3))p3++;
					if (*p3)
					{
						char *p4 = p3;
						while (*p4 && (!white(*p4) && *p4 != ','&&*p4 != ';'))p4++;
						s = *p4; *p4 = 0;
						strcpy(Functions->ins_array->array[Functions->ins_array->filled_elements]->param2, p3);
					}
				}
			}
		}
		Functions->ins_array->filled_elements++;
	}
	return ERR_NO_ERROR;
}

int JVMSimulator::jvmsim_run(string function_name, int line_from, int line_to, int use_files)
{
	if(use_files)
		read_input();
	//locals[0]=101; locals[1]=102;

	max_local_used = 0;
	memset(locals,0,sizeof(int32_t)*MAX_NUMBER_OF_LOCAL_VARIABLES);

	struct Pc PC = { "", 0, NULL };
	strcpy(PC.fn, function_name.c_str()); PC.ln = line_from;
	for (int plus = 0; plus<10; plus++)
	{
		PC.current_ins = find_ins(PC.fn, PC.ln);
		if (PC.current_ins)break;
		PC.ln++;
	}
	if (!PC.current_ins) return (ERR_NO_SUCH_INSTRUCTION);

	int insc = EMULATE_MAX_INSTRUCTIONS;
	do
	{
		int result=emulate_ins(&PC);
		if (!result) break;
		//printf(".");
		insc--;
	} while ((strcmp(PC.fn, function_name.c_str()) || PC.ln<line_to) && insc);
	if(use_files)
		write_output();
	if (!insc)
		return (ERR_MAX_NUMBER_OF_INSTRUCTIONS_EXHAUSTED);
	else
		return (ERR_NO_ERROR);

}

int JVMSimulator::jvmsim_main(int argc, char* argv[])
{
	jvmsim_init();

	if(argc>=2&&!strcmp(argv[1],"getfunctions"))
	{
		// printf("FFMul 62\n");
		
		struct F *fn = Functions;
		while(fn)
		{
			printf("%s ",fn->short_name);
			if(fn->ins_array->filled_elements)
				printf("%i\n",fn->ins_array->array[fn->ins_array->filled_elements-1]->line_number);
			fn=fn->next;
		}
		
		assert(false);
		exit(ERR_NO_ERROR);
	}

	//	for(int i=0;i<800;i++)push(0);

	if(argc>=5&&!strcmp(argv[1],"run"))
	{
		jvmsim_run(argv[2], atoi(argv[3]), atoi(argv[4]),true);
	}

	printf("Usage: %s getfunctions\n       %s run function_name start_line_number end_line_number\n\n",argv[0],argv[0]);
	return ERR_NO_ERROR;
}

string JVMSimulator::getFunctionNameByID(int functionID) {
	struct F *fnct = Functions;

	// If bigger functionID is supplied, scale by modulo
	functionID = functionID % numFunctions;

	int index = 0;
	while (fnct)
	{
		if (index == functionID) {
			return fnct->short_name;
		}
		index++;
		fnct = fnct->next;
	}
	return "NO_SUCH_FUNCTION";
}
