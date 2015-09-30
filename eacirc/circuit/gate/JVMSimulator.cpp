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
struct Instruction *JVMSimulator::find_ins(char *fn, int instructionLineNumber)
{
    struct FunctionNode *fnct= m_functions;
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
    struct CallStackNode *n = static_cast<struct CallStackNode*>(malloc(sizeof(struct CallStackNode)));
    if (n == NULL) {
        mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
        exit(ERR_NO_MEMORY);
    }
    n->next=m_callStack;
    n->function=static_cast<char*>(malloc(strlen(fn)+1));
    if (n->function == NULL) {
        mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
        exit(ERR_NO_MEMORY);
    }
    strcpy(n->function,fn);
    n->next_line=nl;
    m_callStack=n;
}

int JVMSimulator::call_pop(struct Pc *PC)
{
    if(m_callStack==NULL) return 1;
    strcpy(PC->fn,m_callStack->function);
    PC->ln=m_callStack->next_line;
    PC->current_ins=find_ins(PC->fn,PC->ln);
    if (PC->current_ins == NULL) {
        mainLogger.out(LOGGER_WARNING) << "Cannot find instruction during POP operation. Returning 1 (call stack empty)." << endl;
        //exit(ERR_DATA_MISMATCH);
        return 1;
    }
    struct CallStackNode *d=m_callStack;
    m_callStack=m_callStack->next;
    delete d;
    return 0;
}


void JVMSimulator::list_stack()
{
    printf("S: ");
    struct StackNode *n=m_stack;
    while(n)
    {
        printf("-> %i",static_cast<int>(n->integer));
        n=n->next;
    }
    printf("\n");
}

bool JVMSimulator::stack_empty()
{
    return m_stack == NULL;
}

void JVMSimulator::push_int(int32_t ii)
{
    struct StackNode *n=static_cast<struct StackNode*>(malloc(sizeof(struct StackNode)));
    if (n == NULL) {
        mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
        exit(ERR_NO_MEMORY);
    }
    n->next=m_stack;
    n->integer=ii;
    n->data_type=STACKTYPE_INTEGER;
    m_stack=n;
}

void JVMSimulator::push_arrayref(int32_t ii)
{
    struct StackNode *n=static_cast<struct StackNode*>(malloc(sizeof(struct StackNode)));
    if (n == NULL) {
        mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
        exit(ERR_NO_MEMORY);
    }
    n->next=m_stack;
    n->integer=ii;
    n->data_type=STACKTYPE_ARRAYREF;
    m_stack=n;
}

/*
void push(unsigned char c)
{
    struct StackNode *n=static_cast<struct StackNode*>(malloc(sizeof(struct StackNode)));
    if(n==NULL) 
    {
        mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
        exit(ERR_NO_MEMORY);
    }
    n->next=m_stack;
    n->data=c;
    m_stack=n;
}


void push_byte(signed char c)
{
    push((unsigned char)c);
}
*/

int32_t JVMSimulator::pop_int()
{
    if (m_stack == NULL)
    {
        //exit(ERR_STACK_EMPTY);
        //mainLogger.out(LOGGER_WARNING) << "Stack empty during POP. Returning 0." << endl;
        return 0;
    }
    if(m_stack->data_type!=STACKTYPE_INTEGER) {
        //exit(ERR_STACK_DATAMISMATCH);
        mainLogger.out(LOGGER_WARNING) << "Data mismatch during POP. Returning 0." << endl;
        return 0;
    }
    int32_t res = m_stack->integer;
    struct StackNode *d = m_stack;
    m_stack=m_stack->next;
    delete d;
    return res;
}

int32_t JVMSimulator::pop_arrayref()
{
    if (m_stack == NULL) {
        //exit(ERR_STACK_EMPTY);
        //mainLogger.out(LOGGER_WARNING) << "Stack empty during POP ARRAYREF. Returning 0." << endl;
        return 0;
    }
    if (m_stack->data_type != STACKTYPE_ARRAYREF) {
        //exit(ERR_STACK_DATAMISMATCH);
        mainLogger.out(LOGGER_WARNING) << "Data mismatch during POP ARRAYREF. Returning 0." << endl;
        return 0;
    }
    int32_t res = m_stack->integer;
    struct StackNode *d = m_stack;
    m_stack=m_stack->next;
    delete d;
    return res;
}

/*
unsigned char pop()
{
    if(m_stack==NULL) exit(ERR_STACK_EMPTY);
    unsigned char res = m_stack->data;
    struct StackNode *d = m_stack;
    m_stack=m_stack->next;
    delete d;
    return res;
}

signed char pop_byte()
{
    return static_cast<signed char>( pop();
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

    //mainLogger.out(LOGGER_INFO) << "Emulating instruction: " << PC->current_ins->full_line << endl;

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
    case IOR:
    {
                int32_t a = pop_int();
                int32_t b = pop_int();
                int32_t c = a|b;
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
        push_int(static_cast<int32_t>(-1));
        break;
    case ICONST_0:
        push_int(static_cast<int32_t>(0));
        break;
    case ICONST_1:
        push_int(static_cast<int32_t>(1));
        break;
    case ICONST_2:
        push_int(static_cast<int32_t>(2));
        break;
    case ICONST_3:
        push_int(static_cast<int32_t>(3));
        break;
    case ICONST_4:
        push_int(static_cast<int32_t>(4));
        break;
    case ICONST_5:
        push_int(static_cast<int32_t>(5));
        break;
    case I2B:
    {
                int32_t a = pop_int();
                signed char c = static_cast<signed char>(a);
                push_int(static_cast<int>(c));
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
                int32_t c = a >> (b & 0x1f);
                push_int(c);
    }
        break;
    case IUSHR:
    {
                int32_t a = pop_int();
                int32_t b = pop_int();
                int32_t c = static_cast<int32_t>(static_cast<uint32_t>(a) >> (b & 0x1f));
                push_int(c);
    }
        break;
    case BIPUSH:
    {
                int j = atoi(PC->current_ins->param1);
                signed char c = static_cast<signed char>(j);
                push_int(static_cast<int32_t>(c));
    }
        break;
    case SIPUSH:
    {
                int j = atoi(PC->current_ins->param1);
                int16_t s = static_cast<int16_t>(j);
                push_int(static_cast<int32_t>(s));
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
    case IF_ICMPLE:
    {
                    int32_t value2 = pop_int();
                    int32_t value1 = pop_int();
                    if (value1 <= value2)
                    {
                        PC->ln = atoi(PC->current_ins->param1);
                        PC->current_ins = find_ins(PC->fn, PC->ln);
                        if (PC->current_ins == NULL){
                            mainLogger.out(LOGGER_WARNING) << "Wrong IF_ICMPLE jump. Interrupting execution." << endl;
                            //exit(ERR_DATA_MISMATCH);
                            return -1;
                        }
                        jump = 1;
                    }
    }
        break;
    case IF_ICMPEQ:
    {
                    int32_t value2 = pop_int();
                    int32_t value1 = pop_int();
                    if (value1 == value2)
                    {
                        PC->ln = atoi(PC->current_ins->param1);
                        PC->current_ins = find_ins(PC->fn, PC->ln);
                        if (PC->current_ins == NULL){
                            mainLogger.out(LOGGER_WARNING) << "Wrong IF_ICMPEQ jump. Interrupting execution." << endl;
                            //exit(ERR_DATA_MISMATCH);
                            return -1;
                        }
                        jump = 1;
                    }
    }
        break;
    case IF_ICMPGT:
    {
                    int32_t value2 = pop_int();
                    int32_t value1 = pop_int();
                    if (value1 > value2)
                    {
                        PC->ln = atoi(PC->current_ins->param1);
                        PC->current_ins = find_ins(PC->fn, PC->ln);
                        if (PC->current_ins == NULL){
                            mainLogger.out(LOGGER_WARNING) << "Wrong IF_ICMPGT jump. Interrupting execution." << endl;
                            //exit(ERR_DATA_MISMATCH);
                            return -1;
                        }
                        jump = 1;
                    }
    }
        break;
    case IF_ICMPLT:
    {
                    int32_t value2 = pop_int();
                    int32_t value1 = pop_int();
                    if (value1 < value2)
                    {
                        PC->ln = atoi(PC->current_ins->param1);
                        PC->current_ins = find_ins(PC->fn, PC->ln);
                        if (PC->current_ins == NULL){
                            mainLogger.out(LOGGER_WARNING) << "Wrong IF_ICMPLT jump. Interrupting execution." << endl;
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
            struct FunctionNode *fnct = m_functions;
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
    {
                return -1;
    }
        break;
    case ARETURN:
    {
                    if (call_pop(PC) == 1){
                        //exit(ERR_FUNCTION_RETURNED);
                        return 1;
                    }
                    return -1;
    }
        break;
    case IRETURN:
    {
                    if (call_pop(PC) == 1)
                    {
                        // int32_t r=pop_int();
                        // printf("Returning result: %i\n",static_cast<int>(r);
                        write_output();
                        //exit(ERR_FUNCTION_RETURNED);
                        return 1;
                    }

                    return -1;
    }
        break;
    case ILOAD_0:
    {
                    push_int(m_locals[0]);
                    if (m_maxLocalUsed<0)m_maxLocalUsed = 0;
    }
        break;
    case ILOAD_1:
    {
                    push_int(m_locals[1]);
                    if (m_maxLocalUsed<1)m_maxLocalUsed = 1;
    }
        break;
    case ILOAD_2:
    {
                    push_int(m_locals[2]);
                    if (m_maxLocalUsed<2)m_maxLocalUsed = 2;
    }
        break;
    case ILOAD_3:
    {
                    push_int(m_locals[3]);
                    if (m_maxLocalUsed<3)m_maxLocalUsed = 3;
    }
        break;
    case ILOAD:
    {
                // todo: verify index...
                push_int(static_cast<int32_t>(m_locals[atoi(PC->current_ins->param1)]));
                if (m_maxLocalUsed<atoi(PC->current_ins->param1))m_maxLocalUsed = atoi(PC->current_ins->param1);
    }
        break;
    case ISTORE:
    {
                int32_t i = pop_int();
                m_locals[atoi(PC->current_ins->param1)] = i;
                if (m_maxLocalUsed<atoi(PC->current_ins->param1))m_maxLocalUsed = atoi(PC->current_ins->param1);
    }
        break;
    case ISTORE_0:
    {
                    int32_t i = pop_int();
                    m_locals[0] = i;
                    if (m_maxLocalUsed<0)m_maxLocalUsed = 0;
    }
        break;
    case ISTORE_1:
    {
                    int32_t i = pop_int();
                    m_locals[1] = i;
                    if (m_maxLocalUsed<1)m_maxLocalUsed = 1;
    }
        break;
    case ISTORE_2:
    {
                    int32_t i = pop_int();
                    m_locals[2] = i;
                    if (m_maxLocalUsed<2)m_maxLocalUsed = 2;
    }
        break;
    case ISTORE_3:
    {
                    int32_t i = pop_int();
                    m_locals[3] = i;
                    if (m_maxLocalUsed<3)m_maxLocalUsed = 3;
    }
        break;
    case NEWARRAY:
    {
                    int count = pop_int();
                    if (!strcmp(PC->current_ins->param1, "integer"))
                    {
                        m_globalArrays[m_globalArraysCount].int_array = static_cast<int32_t*>(malloc(count*sizeof(int32_t)));
                        if (m_globalArrays[m_globalArraysCount].int_array == NULL) {
                            mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
                            exit(ERR_NO_MEMORY);
                        }
                        m_globalArrays[m_globalArraysCount].type = T_INT;
                        m_globalArrays[m_globalArraysCount].number_of_elements = count;
                        push_arrayref(m_globalArraysCount);
                        m_globalArraysCount++;
                    }
    }
        break;
    case ASTORE_0:
    {
                    int32_t ref = pop_arrayref();
                    m_locals[0] = ref;
                    if (m_maxLocalUsed<0)m_maxLocalUsed = 0;
    }
        break;
    case ASTORE_1:
    {
                    int32_t ref = pop_arrayref();
                    m_locals[1] = ref;
                    if (m_maxLocalUsed<1)m_maxLocalUsed = 1;
    }
        break;
    case ASTORE_2:
    {
                    int32_t ref = pop_arrayref();
                    m_locals[2] = ref;
                    if (m_maxLocalUsed<2)m_maxLocalUsed = 2;
    }
        break;
    case ASTORE_3:
    {
                    int32_t ref = pop_arrayref();
                    m_locals[3] = ref;
                    if (m_maxLocalUsed<3)m_maxLocalUsed = 3;
    }
        break;
    case ASTORE:
    {
                int32_t ref = pop_arrayref();
                // verify index
                m_locals[atoi(PC->current_ins->param1)] = ref;
                if (m_maxLocalUsed<atoi(PC->current_ins->param1))m_maxLocalUsed = atoi(PC->current_ins->param1);
    }
        break;
    case AALOAD:
    {
                int index = pop_int();
                int32_t ref = pop_arrayref();
                if (ref >= m_globalArraysCount) {
                    mainLogger.out(LOGGER_WARNING) << "Array reference is not valid in AALOAD. Interrupting execution." << endl;
                    //exit(ERR_ARRAYREF_NOT_VALID);
                    return -1;
                }
                if (index >= m_globalArrays[ref].number_of_elements) {
                    mainLogger.out(LOGGER_WARNING) << "Array index is out of range in AALOAD. Interrupting execution." << endl;
                    //exit(ERR_ARRAYINDEX_OUT_OF_RANGE);
                    return -1;
                }
                if (m_globalArrays[ref].type == T_INT)
                {
                    push_int(m_globalArrays[ref].int_array[index]);

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
        struct FunctionNode *fnct = m_functions;
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
    if (!strcmp(i, "nop"))return NOP;
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
    if (!strcmp(i, "ior"))return IOR;
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
    if(!strcmp(i,"if_icmpeq"))return IF_ICMPEQ;
    if(!strcmp(i,"if_icmplt"))return IF_ICMPLT;
    if(!strcmp(i,"if_icmpgt"))return IF_ICMPGT;
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
    if(!strcmp(i, "iushr"))return IUSHR;
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

    mainLogger.out(LOGGER_ERROR) << "Instruction not found: '" << i <<  "'." <<  endl;
    assert(false); 
    exit(ERR_DATA_MISMATCH);
}
/*
void printl()
{
printf("Locals - aa: %i, bb: %i, r: %i, t: %i\n", m_locals[2], m_locals[3], m_locals[4], m_locals[5]);
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
            m_locals[local_index] = static_cast<int32_t>(strtol(number, NULL, 16));
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
            push_int(static_cast<int32_t>(strtol(number, NULL, 16)));
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

    struct StackNode *n=m_stack;
    while(n)
    {
        switch(n->data_type)
        {
        case STACKTYPE_INTEGER:
            fprintf(f, "%x ", static_cast<int>(n->integer));
            break;
        case STACKTYPE_ARRAYREF:
            fprintf(f, "%x(ARRAYREF) ", static_cast<int>(n->integer));
            break;
        }
        n=n->next;
    }
    fprintf(f,"\n");
    for(int i=0;i<=m_maxLocalUsed;i++)
        fprintf(f,"%x ", m_locals[i]);
    fclose(f);
}

int JVMSimulator::jvmsim_init()
{
    //FILE *f = fopen("AES.dis", "rt");
    //FILE *f = fopen("oldNodesSimulator.dis", "rt");
    FILE *f = fopen("old_nodes_without_relational_op.dis", "rt");

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
            struct FunctionNode *fnew = (struct FunctionNode*)malloc(sizeof(struct FunctionNode));
            if (fnew == NULL)
            {
                mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
                return(ERR_NO_MEMORY);
            }
            fnew->next = m_functions;
            m_functions = fnew;
            fnew->full_name = static_cast<char*>(malloc(strlen(lastline) + 1));
            if (fnew->full_name == NULL)
            {
                mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
                return(ERR_NO_MEMORY);
            }
            strcpy(fnew->full_name, lastline);
            fnew->short_name = static_cast<char*>(malloc(strlen(lastline) + 1));
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

            fnew->ins_array = static_cast<struct Instructions*>(malloc(sizeof(struct Instruction)));
            if (fnew->ins_array == NULL)
            {
                mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
                return(ERR_NO_MEMORY);
            }
            fnew->ins_array->array = static_cast<struct Instruction**>(malloc(ALLOC_STEP*sizeof(struct Instruction*)));
            fnew->ins_array->filled_elements = 0;
            fnew->ins_array->maximum_elements = ALLOC_STEP;

            m_numFunctions++;
            continue;
        }
        // instructions
        if (m_functions == NULL)
            return(ERR_DATA_MISMATCH);
        if (m_functions->ins_array->filled_elements >= m_functions->ins_array->maximum_elements)
        {
            m_functions->ins_array->array = static_cast<struct Instruction**>(realloc(m_functions->ins_array->array, (m_functions->ins_array->maximum_elements + ALLOC_STEP)*sizeof(struct Instruction*)));
            if (m_functions->ins_array->array == NULL)
            {
                mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
                return(ERR_NO_MEMORY);
            }
            m_functions->ins_array->maximum_elements += ALLOC_STEP;
        }
        m_functions->ins_array->array[m_functions->ins_array->filled_elements] = static_cast<struct Instruction*>(malloc(sizeof(struct Instruction)));
        if (m_functions->ins_array->array[m_functions->ins_array->filled_elements] == NULL)
        {
            mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
            return(ERR_NO_MEMORY);
        }
        m_functions->ins_array->array[m_functions->ins_array->filled_elements]->full_line = static_cast<char*>(malloc(strlen(line) + 1));
        if (m_functions->ins_array->array[m_functions->ins_array->filled_elements]->full_line == NULL)
        {
            mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
            return(ERR_NO_MEMORY);
        }
        strcpy(m_functions->ins_array->array[m_functions->ins_array->filled_elements]->full_line, line);
        m_functions->ins_array->array[m_functions->ins_array->filled_elements]->line_number = atoi(line);
        m_functions->ins_array->array[m_functions->ins_array->filled_elements]->instruction = static_cast<char*>(malloc(strlen(line) + 1));
        if (m_functions->ins_array->array[m_functions->ins_array->filled_elements]->instruction == NULL)
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
        strcpy(m_functions->ins_array->array[m_functions->ins_array->filled_elements]->instruction, p1);
        m_functions->ins_array->array[m_functions->ins_array->filled_elements]->instruction_code = code(m_functions->ins_array->array[m_functions->ins_array->filled_elements]->instruction);
        m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param1 = static_cast<char*>(malloc(strlen(line) + 1));
        if (m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param1 == NULL)
        {
            mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
            return(ERR_NO_MEMORY);
        }
        strcpy(m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param1, "");
        m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param2 = static_cast<char*>(malloc(strlen(line) + 1));
        if (m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param2 == NULL)
        {
            mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
            return(ERR_NO_MEMORY);
        }
        strcpy(m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param2, "");
        *p2 = s;
        if (*p2)
        {
            while (*p2&&white(*p2))p2++;
            if (*p2)
            {
                char *p3 = p2;
                while (*p3 && (!white(*p3) && *p3 != ','&&*p3 != ';'))p3++;
                s = *p3; *p3 = 0;
                strcpy(m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param1, p2);
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
                        strcpy(m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param2, p3);
                    }
                }
            }
        }
        m_functions->ins_array->filled_elements++;
    }
    return ERR_NO_ERROR;
}

int JVMSimulator::jvmsim_run(int function_number, int line_from, int line_to, int use_files)
{
    if(use_files)
        read_input();
    //m_locals[0]=101; m_locals[1]=102;

    struct FunctionNode* function = get_function_by_number(function_number);
    if (function == NULL){
        mainLogger.out(LOGGER_ERROR) << "Cannot find function with nubmer: " << function_number << endl;
        return (ERR_NO_SUCH_FUNCTION);
    }

    //mainLogger.out(LOGGER_INFO) << "Running function: " << function->short_name << "[" << line_from << ", " << line_to  << "]" << endl;

    m_maxLocalUsed = 0;
    memset(m_locals,0,sizeof(int32_t)*MAX_NUMBER_OF_LOCAL_VARIABLES);

    struct Pc PC = { "", 0, NULL };
    strcpy(PC.fn, function->short_name); PC.ln = line_from;
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
        int result = emulate_ins(&PC);
        if (result == -1) break;
        //printf(".");
        insc--;
    } while ((strcmp(PC.fn, function->short_name) || PC.ln<=line_to) && insc);

    if(use_files)
        write_output();
    if (!insc)
        return (ERR_MAX_NUMBER_OF_INSTRUCTIONS_EXHAUSTED);
    else
        return (ERR_NO_ERROR);
}

int JVMSimulator::get_num_of_functions(){
    int ret = 0;
    struct FunctionNode* current = m_functions;

    while (current != NULL){
        ret++;
        current = current->next;
    }

    return ret;
}

struct FunctionNode* JVMSimulator::get_function_by_number(unsigned char num){
    struct FunctionNode* current = m_functions;

    for (unsigned char i = 0; i < num; i++){
        current = current->next;
        if (current == NULL){
            return NULL;
        }
    }

    return current;
}

/*
int JVMSimulator::jvmsim_main(int argc, char* argv[])
{
    jvmsim_init();

    if(argc>=2&&!strcmp(argv[1],"getfunctions"))
    {
        // printf("FFMul 62\n");
        
        struct FunctionNode *fn = m_functions;
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

    //for(int i=0;i<800;i++)push(0);

    if(argc>=5&&!strcmp(argv[1],"run"))
    {
        jvmsim_run(argv[2], atoi(argv[3]), atoi(argv[4]),true);
    }

    printf("Usage: %s getfunctions\n       %s run function_name start_line_number end_line_number\n\n",argv[0],argv[0]);
    return ERR_NO_ERROR;
}*/

/*string JVMSimulator::getFunctionNameByID(int functionID) {
    struct FunctionNode *fnct = m_functions;

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
}*/
