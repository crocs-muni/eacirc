#include "JVMSimulator.h"
#include "EACglobals.h"
#include <cassert>

#pragma warning(disable:4996)

JVMSimulator::JVMSimulator() {
    int retval = jvmsim_init(pGlobals->settings->gateCircuit.jvmSimFilename);
    if (retval == 0) {
        mainLogger.out(LOGGER_INFO) << "JVM simulator initialized. Loaded file: " <<
        pGlobals->settings->gateCircuit.jvmSimFilename << endl;
    } else {
        mainLogger.out(LOGGER_ERROR) << "JVM simulator initialisation returned error (" << retval << ")" << endl;
    }

    for (int i = 0; i < MAX_NUMBER_OF_VARIABLES; i++) {
        m_globalArraysUsed[i] = false;
        m_localsUsed[i] = false;
    }
}

JVMSimulator::~JVMSimulator() { }

void JVMSimulator::call_push(struct Pc* PC) {
    struct CallStackNode* n = static_cast<struct CallStackNode*>(malloc(sizeof(struct CallStackNode)));
    if (n == NULL) {
        mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
        exit(ERR_NO_MEMORY);
    }

    n->next = m_callStack;
    n->function = PC->currentFunction;
    n->next_line = PC->ln;
    m_callStack = n;
}

int JVMSimulator::call_pop(struct Pc* PC) {
    if (m_callStack == NULL) return 1;

    if (m_callStack->next_line >= m_callStack->function->ins_array->filled_elements) {
        mainLogger.out(LOGGER_WARNING) <<
        "Cannot find instruction during POP operation. Returning 1 (call stack empty)." << endl;
        return 1;
    }

    PC->currentFunction = m_callStack->function;
    PC->ln = m_callStack->next_line;

    struct CallStackNode* d = m_callStack;
    m_callStack = m_callStack->next;
    delete d;
    return 0;
}


void JVMSimulator::list_stack() {
    mainLogger.out(LOGGER_INFO) << "S: ";
    struct StackNode* n = m_stack;
    while (n) {
        mainLogger.out(LOGGER_INFO) << " -> " << static_cast<int>(n->integer);
        n = n->next;
    }
    mainLogger.out(LOGGER_INFO) << endl;
}

bool JVMSimulator::stack_empty() {
    return m_stack == NULL;
}

bool JVMSimulator::callStackEmpty() {
    return m_callStack == NULL;
}

void JVMSimulator::push_int(int32_t value) {
    struct StackNode* n = static_cast<struct StackNode*>(malloc(sizeof(struct StackNode)));
    if (n == NULL) {
        mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
        exit(ERR_NO_MEMORY);
    }

    n->next = m_stack;
    n->integer = value;
    n->data_type = STACKTYPE_INTEGER;
    m_stack = n;
}

void JVMSimulator::push_long(int64_t value) {
    push_int(static_cast<int32_t>(value));
    push_int(static_cast<int32_t>(value >> 32));
}

void JVMSimulator::push_arrayref(int32_t ii) {
    struct StackNode* n = static_cast<struct StackNode*>(malloc(sizeof(struct StackNode)));
    if (n == NULL) {
        mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
        exit(ERR_NO_MEMORY);
    }

    n->next = m_stack;
    n->integer = ii;
    n->data_type = STACKTYPE_ARRAYREF;
    m_stack = n;
}

void JVMSimulator::pop_int(int32_t& returnValue) {
    if (m_stack == NULL) {
        returnValue = 0;
        return;
    }

    if (m_stack->data_type != STACKTYPE_INTEGER) {
        returnValue = 0;
        return;
    }

    returnValue = m_stack->integer;
    struct StackNode* d = m_stack;
    m_stack = m_stack->next;
    delete d;
    //return; // STACK_ERR_NO_ERROR;
}

void JVMSimulator::pop_long(int64_t& returnValue) {
    int32_t word1;
    int32_t word2;

    pop_int(word1);
    pop_int(word2);

    returnValue = createLongFromTwoInts(word1, word2);
}


void JVMSimulator::pop_arrayref(int32_t& returnValue) {
    if (m_stack == NULL) {
        returnValue = 0;
        return;
    }

    if (m_stack->data_type != STACKTYPE_ARRAYREF) {
        returnValue = 0;
        return;
    }

    returnValue = m_stack->integer;
    struct StackNode* d = m_stack;
    m_stack = m_stack->next;
    delete d;
}

int JVMSimulator::emulate_ins(struct Pc* PC, int& returnValue) {
    //mainLogger.out(LOGGER_INFO) << "Emulating instruction: " << PC->currentFunction->ins_array->array[PC->ln]->full_line << endl;

    switch (PC->currentFunction->ins_array->array[PC->ln]->instruction_code) {
        case NOP: {
            break;
        }
        case LCONST_0: {
            push_long(0);
        }
            break;
        case LCONST_1: {
            push_long(1);
        }
            break;
        case IADD: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t a;
            int32_t b;
            pop_int(a);
            pop_int(b);
            int32_t c = a + b;
            push_int(c);
        }
            break;
        case LADD: {
            if (!hasFourIntegersOnStack()) {
                return CONTINUE;
            }
            int64_t a;
            int64_t b;
            pop_long(a);
            pop_long(b);
            push_long(a + b);
        }
            break;
        case ISUB: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t a;
            int32_t b;
            pop_int(a);
            pop_int(b);
            int32_t c = b - a;
            push_int(c);
        }
            break;
        case LSUB: {
            if (!hasFourIntegersOnStack()) {
                return CONTINUE;
            }
            int64_t a;
            int64_t b;
            pop_long(a);
            pop_long(b);
            push_long(b - a);
        }
            break;
        case IAND: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t a;
            int32_t b;
            pop_int(a);
            pop_int(b);
            int32_t c = a & b;
            push_int(c);
        }
            break;
        case LAND: {
            if (!hasFourIntegersOnStack()) {
                return CONTINUE;
            }
            int64_t a;
            int64_t b;
            pop_long(a);
            pop_long(b);
            push_long(a & b);
        }
            break;
        case IOR: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t a;
            int32_t b;
            pop_int(a);
            pop_int(b);
            int32_t c = a | b;
            push_int(c);
        }
            break;
        case IXOR: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t a;
            int32_t b;
            pop_int(a);
            pop_int(b);
            int32_t c = a ^b;
            push_int(c);
        }
            break;
        case I2L: {
            if (!hasIntegerOnStack()) {
                return CONTINUE;
            }
            int32_t a;
            pop_int(a);
            push_long(static_cast<int64_t>(a));
        }
            break;
        case L2I: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int64_t a;
            pop_long(a);
            push_int(static_cast<int32_t>(a));
        }
            break;
        case I2S: {
            if (!hasIntegerOnStack()) {
                return CONTINUE;
            }
            int32_t a;
            pop_int(a);
            int16_t c = static_cast<int16_t>(a);
            push_int(static_cast<int32_t>(c));
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
        case I2B: {
            if (!hasIntegerOnStack()) {
                return CONTINUE;
            }
            int32_t a;
            pop_int(a);
            signed char c = static_cast<signed char>(a);
            push_int(static_cast<int32_t>(c));
        }
            break;
        case I2C: {
            if (!hasIntegerOnStack()) {
                return CONTINUE;
            }
            int32_t a;
            pop_int(a);
            int16_t c = static_cast<int16_t>(a);
            a = 0 + c;
            push_int(a);
        }
            break;
        case IDIV: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t a;
            int32_t b;
            pop_int(a);
            pop_int(b);
            if (a == 0) {
                push_int(a);
                push_int(b);
                return CONTINUE;
            }
            int32_t c = b / a;
            push_int(c);
        }
            break;
        case LDIV: {
            if (!hasFourIntegersOnStack()) {
                return CONTINUE;
            }

            int64_t value1;
            int64_t value2;

            pop_long(value1);

            if (value1 == 0) {
                push_long(value1);
                return CONTINUE;
            }

            pop_long(value2);
            push_long(value2 / value1);
        }
            break;
        case IMUL: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t a;
            int32_t b;
            pop_int(a);
            pop_int(b);
            int32_t c = b * a;
            push_int(c);
        }
            break;
        case IREM: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t a;
            int32_t b;
            pop_int(a);
            pop_int(b);
            if (a == 0) {
                push_int(a);
                push_int(b);
                return CONTINUE;
            }
            int32_t c = b % a;
            push_int(c);
        }
            break;
        case ISHL: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t a;
            int32_t b;
            pop_int(a);
            pop_int(b);
            int32_t c = b << (a & 0x1f);
            push_int(c);
        }
            break;
        case ISHR: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t a;
            int32_t b;
            pop_int(a);
            pop_int(b);
            int32_t c = a >> (b & 0x1f);
            push_int(c);
        }
            break;
        case IUSHR: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t a;
            int32_t b;
            pop_int(a);
            pop_int(b);
            int32_t c = static_cast<int32_t>(static_cast<uint32_t>(a) >> (b & 0x1f));
            push_int(c);
        }
            break;
        case LSHR: {
            if (!hasThreeIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t value1;
            int64_t value2;

            pop_int(value1);
            pop_long(value2);

            value2 = value2 >> (value1 & 0x3f);

            push_long(value2);
        }
            break;
        case BIPUSH: {
            int j = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);
            signed char c = static_cast<signed char>(j);
            push_int(static_cast<int32_t>(c));
        }
            break;
        case SIPUSH: {
            int j = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);
            int16_t s = static_cast<int16_t>(j);
            push_int(static_cast<int32_t>(s));
        }
            break;
        case DUP2: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int64_t value;
            pop_long(value);

            push_long(value);
            push_long(value);
        }
            break;
        case DUP: {
            if (!hasIntegerOnStack()) {
                return CONTINUE;
            }
            int32_t j;
            pop_int(j);
            push_int(j);
            push_int(j);
        }
            break;
        case GOTO: {
            returnValue = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);
            return GOTO_VALUE;
        }
            break;
        case LCMP: {
            if (!hasFourIntegersOnStack()) {
                return CONTINUE;
            }

            int64_t value1;
            int64_t value2;

            pop_long(value1);
            pop_long(value2);

            if (value1 == value2) {
                push_int(0);
                return CONTINUE;
            }

            if (value2 > value1) {
                push_int(1);
            } else {
                push_int(-1);
            }
        }
            break;
        case IFEQ: {
            if (!hasIntegerOnStack()) {
                return CONTINUE;
            }
            int32_t value;
            pop_int(value);
            if (value == 0) {
                returnValue = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);
                return GOTO_VALUE;
            }
        }
            break;
        case IFGE: {
            if (!hasIntegerOnStack()) {
                return CONTINUE;
            }
            int32_t value;
            pop_int(value);
            if (value >= 0) {
                returnValue = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);
                return GOTO_VALUE;
            }
        }
            break;
        case IFLE: {
            if (!hasIntegerOnStack()) {
                return CONTINUE;
            }
            int32_t value;
            pop_int(value);
            if (value <= 0) {
                returnValue = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);
                return GOTO_VALUE;
            }
        }
            break;
        case IF_ICMPGE: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t value2;
            int32_t value1;
            pop_int(value2);
            pop_int(value1);
            if (value1 >= value2) {
                returnValue = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);
                return GOTO_VALUE;
            }
        }
            break;
        case IF_ICMPLE: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t value2;
            int32_t value1;
            pop_int(value2);
            pop_int(value1);
            if (value1 <= value2) {
                returnValue = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);
                return GOTO_VALUE;
            }
        }
            break;
        case IF_ICMPEQ: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t value2;
            int32_t value1;
            pop_int(value2);
            pop_int(value1);
            if (value1 == value2) {
                returnValue = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);
                return GOTO_VALUE;
            }
        }
            break;
        case IF_ICMPGT: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t value2;
            int32_t value1;
            pop_int(value2);
            pop_int(value1);
            if (value1 > value2) {
                returnValue = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);
                return GOTO_VALUE;
            }
        }
            break;
        case IF_ICMPLT: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }
            int32_t value2;
            int32_t value1;
            pop_int(value2);
            pop_int(value1);
            if (value1 < value2) {
                returnValue = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);
                return GOTO_VALUE;
            }
        }
            break;
        case IINC: {
            uint16_t varnum = static_cast<uint16_t>(atoi(PC->currentFunction->ins_array->array[PC->ln]->param1));
            int16_t n = static_cast<int16_t>(atoi(PC->currentFunction->ins_array->array[PC->ln]->param1));

            if (varnum >= MAX_NUMBER_OF_VARIABLES || !m_localsUsed[varnum]) {
                return CONTINUE;
            }

            m_locals[varnum] += n;
        }
        case LDC:
        case LDC_W: {
            push_int(atoi(PC->currentFunction->ins_array->array[PC->ln]->param1));
        }
            break;
        case LDC2_W: {
            push_long(atol(PC->currentFunction->ins_array->array[PC->ln]->param1));
        }
            break;
        case INVOKEVIRTUAL:
        case INVOKESTATIC: {
            if (PC->currentFunction->ins_array->filled_elements <= PC->ln) {
                mainLogger.out(LOGGER_WARNING) << "Data mismatch during INVOKESTATIC jump. Interrupting execution." <<
                endl;
                return ERR_DATA_MISMATCH;
            }

            call_push(PC);

            char cp[MAX_LINE_LENGTH], * p1, * p2;
            strcpy(cp, PC->currentFunction->ins_array->array[PC->ln]->full_line);
            p1 = strstr(cp, "//Method ");
            if (!p1) {
                mainLogger.out(LOGGER_WARNING) << "Data mismatch during INVOKESTATIC jump. Interrupting execution." <<
                endl;
                return -1;
            }
            p1 += 9;
            p2 = strchr(p1, ':');
            if (!p2) {
                mainLogger.out(LOGGER_WARNING) << "Data mismatch during INVOKESTATIC jump. Interrupting execution." <<
                endl;
                return -1;
            }
            *p2 = 0;
            FunctionNode* nextFunction = get_function_by_name(p1);
            if (nextFunction == NULL) {
                mainLogger.out(LOGGER_WARNING) << "Data mismatch during INVOKESTATIC jump. Interrupting execution." <<
                endl;
                return ERR_DATA_MISMATCH;
            }
            if (nextFunction->ins_array->filled_elements == 0) {
                return ERR_DATA_MISMATCH;
            }

            PC->currentFunction = nextFunction;
            PC->ln = 0;
        }
            break;
        case RETURN:
        case ARETURN:
        case IRETURN: {
            call_pop(PC);
        }
            break;
        case LREM: {
            if (!hasFourIntegersOnStack()) {
                return CONTINUE;
            }

            int64_t value1;
            int64_t value2;

            pop_long(value1);

            if (value1 == 0) {
                push_long(value1);
                return CONTINUE;
            }
            pop_long(value2);

            push_long(value2 % value1);
        }
            break;
        case ILOAD_0: {
            if (!m_localsUsed[0]) {
                return CONTINUE;
            }
            push_int(m_locals[0]);
            m_locals[0] = false;
        }
            break;
        case ILOAD_1: {
            if (!m_localsUsed[1]) {
                return CONTINUE;
            }
            push_int(m_locals[1]);
            m_locals[1] = false;
        }
            break;
        case ILOAD_2: {
            if (!m_localsUsed[2]) {
                return CONTINUE;
            }
            push_int(m_locals[2]);
            m_locals[2] = false;
        }
            break;
        case ILOAD_3: {
            if (!m_localsUsed[3]) {
                return CONTINUE;
            }
            push_int(m_locals[3]);
            m_locals[3] = false;
        }
            break;
        case ILOAD: {
            int index = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);

            if (index >= MAX_NUMBER_OF_VARIABLES || !m_localsUsed[index]) {
                return CONTINUE;
            }

            push_int(static_cast<int32_t>(m_locals[index]));
            m_localsUsed[index] = false;
        }
            break;
        case LLOAD_0: {
            if (!m_localsUsed[0] || !m_localsUsed[1]) {
                return CONTINUE;
            }
            push_int(m_locals[1]);
            m_locals[1] = false;

            push_int(m_locals[0]);
            m_locals[0] = false;
        }
            break;
        case LLOAD_1: {
            if (!m_localsUsed[1] || !m_localsUsed[2]) {
                return CONTINUE;
            }
            push_int(m_locals[2]);
            m_locals[2] = false;

            push_int(m_locals[1]);
            m_locals[1] = false;
        }
            break;
        case LLOAD_2: {
            if (!m_localsUsed[2] || !m_localsUsed[3]) {
                return CONTINUE;
            }
            push_int(m_locals[3]);
            m_locals[3] = false;

            push_int(m_locals[2]);
            m_locals[2] = false;
        }
            break;
        case LLOAD_3: {
            if (!m_localsUsed[3] || !m_localsUsed[4]) {
                return CONTINUE;
            }
            push_int(m_locals[4]);
            m_locals[4] = false;

            push_int(m_locals[3]);
            m_locals[3] = false;
        }
            break;
        case LLOAD: {
            int index = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);

            if (index + 1 >= MAX_NUMBER_OF_VARIABLES || !m_localsUsed[index] || !m_localsUsed[index + 1]) {
                return CONTINUE;
            }

            push_int(m_locals[index + 1]);
            m_localsUsed[index + 1] = false;

            push_int(m_locals[index]);
            m_localsUsed[index] = false;
        }
            break;
        case ISTORE: {
            if (!hasIntegerOnStack()) {
                return CONTINUE;
            }
            int index = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);

            if (index >= MAX_NUMBER_OF_VARIABLES) {
                return CONTINUE;
            }
            int32_t i;
            pop_int(i);
            m_locals[index] = i;
            m_localsUsed[index] = true;
        }
            break;
        case ISTORE_0: {
            if (!hasIntegerOnStack()) {
                return CONTINUE;
            }
            int32_t i;
            pop_int(i);
            m_locals[0] = i;
            m_localsUsed[0] = true;
        }
            break;
        case ISTORE_1: {
            if (!hasIntegerOnStack()) {
                return CONTINUE;
            }
            int32_t i;
            pop_int(i);
            m_locals[1] = i;
            m_localsUsed[1] = true;
        }
            break;
        case ISTORE_2: {
            if (!hasIntegerOnStack()) {
                return CONTINUE;
            }
            int32_t i;
            pop_int(i);
            m_locals[2] = i;
            m_localsUsed[2] = true;
        }
            break;
        case ISTORE_3: {
            if (!hasIntegerOnStack()) {
                return CONTINUE;
            }
            int32_t i;
            pop_int(i);
            m_locals[3] = i;
            m_localsUsed[3] = true;
        }
            break;
        case NEWARRAY: {
            if (!hasIntegerOnStack()) {
                return CONTINUE;
            }
            int count;
            pop_int(count);

            if (!strcmp(PC->currentFunction->ins_array->array[PC->ln]->param1, "integer")) {
                int next_global_index = get_next_global_index();

                if (next_global_index == -1) {
                    mainLogger.out(LOGGER_ERROR) << "Memory for global arrays exhausted. Skipping." << endl;
                    return CONTINUE;
                }

                m_globalArrays[next_global_index].int_array = static_cast<int32_t*>(malloc(count * sizeof(int32_t)));
                if (m_globalArrays[next_global_index].int_array == NULL) {
                    mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory. Exiting..." << endl;
                    exit(ERR_NO_MEMORY);
                }

                m_globalArrays[next_global_index].type = T_INT;
                m_globalArrays[next_global_index].number_of_elements = count;
                push_arrayref(next_global_index);
                m_globalArraysUsed[next_global_index] = true;
            }
        }
            break;
        case ASTORE_0: {
            if (!hasArrayRefOnStack()) {
                return CONTINUE;
            }
            int32_t ref;
            pop_arrayref(ref);
            m_locals[0] = ref;
            m_localsUsed[0] = true;
        }
            break;
        case ASTORE_1: {
            if (!hasArrayRefOnStack()) {
                return CONTINUE;
            }
            int32_t ref;
            pop_arrayref(ref);
            m_locals[1] = ref;
            m_localsUsed[1] = true;
        }
            break;
        case ASTORE_2: {
            if (!hasArrayRefOnStack()) {
                return CONTINUE;
            }
            int32_t ref;
            pop_arrayref(ref);
            m_locals[2] = ref;
            m_localsUsed[2] = true;
        }
            break;
        case ASTORE_3: {
            if (!hasArrayRefOnStack()) {
                return CONTINUE;
            }
            int32_t ref;
            pop_arrayref(ref);
            m_locals[3] = ref;
            m_localsUsed[3] = true;
        }
        case LSTORE_0: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }

            int32_t first;
            int32_t second;

            pop_int(first);
            pop_int(second);


            m_locals[0] = second;
            m_localsUsed[0] = true;
            m_locals[1] = first;
            m_localsUsed[1] = true;
        }
            break;
        case LSTORE_1: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }

            int32_t first;
            int32_t second;

            pop_int(first);
            pop_int(second);

            m_locals[1] = second;
            m_localsUsed[1] = true;
            m_locals[2] = first;
            m_localsUsed[2] = true;
        }
            break;
        case LSTORE_2: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }

            int32_t first;
            int32_t second;

            pop_int(second);
            pop_int(first);

            m_locals[2] = second;
            m_localsUsed[2] = true;
            m_locals[3] = first;
            m_localsUsed[3] = true;
        }
            break;
        case LSTORE_3: {
            if (!hasTwoIntegersOnStack()) {
                return CONTINUE;
            }

            int32_t first;
            int32_t second;

            pop_int(second);
            pop_int(first);

            m_locals[3] = second;
            m_localsUsed[3] = true;
            m_locals[4] = first;
            m_localsUsed[4] = true;
        }
            break;
        case LSTORE: {
            int index = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);

            if (index + 1 >= MAX_NUMBER_OF_VARIABLES || !hasTwoIntegersOnStack()) {
                return CONTINUE;
            }

            int32_t first;
            int32_t second;

            pop_int(second);
            pop_int(first);


            m_locals[index] = second;
            m_localsUsed[index] = true;
            m_locals[index + 1] = first;
            m_localsUsed[index + 1] = true;
        }
            break;
        case ASTORE: {
            int index = atoi(PC->currentFunction->ins_array->array[PC->ln]->param1);
            if (index >= MAX_NUMBER_OF_VARIABLES || !hasArrayRefOnStack()) {
                return CONTINUE;
            }

            int32_t ref;
            pop_arrayref(ref);
            m_locals[index] = ref;
            m_localsUsed[index] = true;
        }
            break;
        case AALOAD: {
            if (!hasIntegerOnStack()) {
                return CONTINUE;
            }
            int index;
            int32_t ref;

            pop_int(index);
            if (!hasArrayRefOnStack()) {
                push_int(index);
                return CONTINUE;
            }
            pop_arrayref(ref);
            if (!m_globalArraysUsed[ref]) {
                push_int(index);
                push_arrayref(ref);
                mainLogger.out(LOGGER_WARNING) << "Array reference is not valid in AALOAD. Interrupting execution." <<
                endl;
                return CONTINUE;
            }
            if (index >= m_globalArrays[ref].number_of_elements) {
                mainLogger.out(LOGGER_WARNING) << "Array index is out of range in AALOAD. Interrupting execution." <<
                endl;
                return CONTINUE;
            }
            if (m_globalArrays[ref].type == T_INT) {
                push_int(m_globalArrays[ref].int_array[index]);
            } else {
                /*TODO: handle also other data types */
                mainLogger.out(LOGGER_WARNING) <<
                "This data type is not implemented for AALOAD. Interrupting execution." << endl;
                return CONTINUE;
            }
        }
            break;
        default:
            break;
    }

    return CONTINUE;
}

int JVMSimulator::white(char c) {
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') return 1;
    return 0;
}

int JVMSimulator::code(char* i) {
    if (!strcmp(i, "nop"))return NOP;
    if (!strcmp(i, "lconst_0"))return LCONST_0;
    if (!strcmp(i, "lconst_1"))return LCONST_1;
    if (!strcmp(i, "aaload"))return AALOAD;
    if (!strcmp(i, "aastore"))return AASTORE;
    if (!strcmp(i, "aload"))return ALOAD;
    if (!strcmp(i, "aload_0"))return ALOAD_0;
    if (!strcmp(i, "aload_1"))return ALOAD_1;
    if (!strcmp(i, "aload_2"))return ALOAD_2;
    if (!strcmp(i, "aload_3"))return ALOAD_3;
    if (!strcmp(i, "areturn"))return ARETURN;
    if (!strcmp(i, "arraylength"))return ARRAYLENGTH;
    if (!strcmp(i, "lstore"))return LSTORE;
    if (!strcmp(i, "astore"))return ASTORE;
    if (!strcmp(i, "lstore_0"))return LSTORE_0;
    if (!strcmp(i, "lstore_1"))return LSTORE_1;
    if (!strcmp(i, "lstore_2"))return LSTORE_2;
    if (!strcmp(i, "lstore_3"))return LSTORE_3;
    if (!strcmp(i, "astore_0"))return ASTORE_0;
    if (!strcmp(i, "astore_1"))return ASTORE_1;
    if (!strcmp(i, "astore_2"))return ASTORE_2;
    if (!strcmp(i, "astore_3"))return ASTORE_3;
    if (!strcmp(i, "baload"))return BALOAD;
    if (!strcmp(i, "caload"))return CALOAD;
    if (!strcmp(i, "bastore"))return BASTORE;
    if (!strcmp(i, "castore"))return CASTORE;
    if (!strcmp(i, "bipush"))return BIPUSH;
    if (!strcmp(i, "dup2"))return DUP2;
    if (!strcmp(i, "dup"))return DUP;
    if (!strcmp(i, "getstatic"))return GETSTATIC;
    if (!strcmp(i, "goto"))return GOTO;
    if (!strcmp(i, "i2b"))return I2B;
    if (!strcmp(i, "i2c"))return I2C;
    if (!strcmp(i, "iadd"))return IADD;
    if (!strcmp(i, "iand"))return IAND;
    if (!strcmp(i, "ior"))return IOR;
    if (!strcmp(i, "iaload"))return IALOAD;
    if (!strcmp(i, "iastore"))return IASTORE;
    if (!strcmp(i, "iconst_m1"))return ICONST_M1;
    if (!strcmp(i, "iconst_0"))return ICONST_0;
    if (!strcmp(i, "iconst_1"))return ICONST_1;
    if (!strcmp(i, "iconst_2"))return ICONST_2;
    if (!strcmp(i, "iconst_3"))return ICONST_3;
    if (!strcmp(i, "iconst_4"))return ICONST_4;
    if (!strcmp(i, "iconst_5"))return ICONST_5;
    if (!strcmp(i, "idiv"))return IDIV;
    if (!strcmp(i, "ldiv"))return LDIV;
    if (!strcmp(i, "lcmp"))return LCMP;
    if (!strcmp(i, "ifeq"))return IFEQ;
    if (!strcmp(i, "ifne"))return IFNE;
    if (!strcmp(i, "ifge"))return IFGE;
    if (!strcmp(i, "if_icmpge"))return IF_ICMPGE;
    if (!strcmp(i, "if_icmple"))return IF_ICMPLE;
    if (!strcmp(i, "if_acmpeq"))return IF_ACMPEQ;
    if (!strcmp(i, "if_icmpne"))return IF_ICMPNE;
    if (!strcmp(i, "if_icmpeq"))return IF_ICMPEQ;
    if (!strcmp(i, "if_icmplt"))return IF_ICMPLT;
    if (!strcmp(i, "if_icmpgt"))return IF_ICMPGT;
    if (!strcmp(i, "iinc"))return IINC;
    if (!strcmp(i, "ldc"))return LDC;
    if (!strcmp(i, "ldc_w"))return LDC_W;
    if (!strcmp(i, "ldc2_w"))return LDC2_W;
    if (!strcmp(i, "iload"))return ILOAD;
    if (!strcmp(i, "lload"))return LLOAD;
    if (!strcmp(i, "iload_0"))return ILOAD_0;
    if (!strcmp(i, "iload_1"))return ILOAD_1;
    if (!strcmp(i, "iload_2"))return ILOAD_2;
    if (!strcmp(i, "iload_3"))return ILOAD_3;
    if (!strcmp(i, "lload_0"))return LLOAD_0;
    if (!strcmp(i, "lload_1"))return LLOAD_1;
    if (!strcmp(i, "lload_2"))return LLOAD_2;
    if (!strcmp(i, "lload_3"))return LLOAD_3;
    if (!strcmp(i, "imul"))return IMUL;
    if (!strcmp(i, "getfield"))return GETFIELD;
    if (!strcmp(i, "putfield"))return PUTFIELD;
    if (!strcmp(i, "invokespecial"))return INVOKESPECIAL;
    if (!strcmp(i, "invokevirtual"))return INVOKEVIRTUAL;
    if (!strcmp(i, "invokestatic"))return INVOKESTATIC;
    if (!strcmp(i, "irem"))return IREM;
    if (!strcmp(i, "ireturn"))return IRETURN;
    if (!strcmp(i, "lrem"))return LREM;
    if (!strcmp(i, "ishl"))return ISHL;
    if (!strcmp(i, "ishr"))return ISHR;
    if (!strcmp(i, "lshr"))return LSHR;
    if (!strcmp(i, "iushr"))return IUSHR;
    if (!strcmp(i, "land"))return LAND;
    if (!strcmp(i, "istore"))return ISTORE;
    if (!strcmp(i, "istore_0"))return ISTORE_0;
    if (!strcmp(i, "istore_1"))return ISTORE_1;
    if (!strcmp(i, "istore_2"))return ISTORE_2;
    if (!strcmp(i, "istore_3"))return ISTORE_3;
    if (!strcmp(i, "ladd"))return LADD;
    if (!strcmp(i, "isub"))return ISUB;
    if (!strcmp(i, "lsub"))return LSUB;
    if (!strcmp(i, "ixor"))return IXOR;
    if (!strcmp(i, "i2l"))return I2L;
    if (!strcmp(i, "l2i"))return L2I;
    if (!strcmp(i, "i2s"))return I2S;
    if (!strcmp(i, "ifle"))return IFLE;
    if (!strcmp(i, "multianewarray"))return MULTIANEWARRAY;
    if (!strcmp(i, "newarray"))return NEWARRAY;
    if (!strcmp(i, "pop"))return POP;
    if (!strcmp(i, "putstatic"))return PUTSTATIC;
    if (!strcmp(i, "new"))return NEW;
    if (!strcmp(i, "return"))return RETURN;
    if (!strcmp(i, "sipush"))return SIPUSH;

    mainLogger.out(LOGGER_ERROR) << "Instruction not found: '" << i << "'." << endl;
    assert(false);
    exit(ERR_DATA_MISMATCH);
}

int JVMSimulator::jvmsim_init(string fileName) {
    mainLogger.out(LOGGER_INFO) << "JVMSimulator loading file: " << fileName << endl;
    FILE* f = fopen(fileName.c_str(), "rt");

    if (f == NULL) return (ERR_CANNOT_OPEN_FILE);
    char line[MAX_LINE_LENGTH], lastline[MAX_LINE_LENGTH];

    // read lines and parse
    strcpy(line, "");
    while (strcpy(lastline, line), fgets(line, MAX_LINE_LENGTH, f) != NULL) {
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
        if (!strncmp(line, "  Code:", strlen("Code:"))) {
            // new function
            struct FunctionNode* fnew = (struct FunctionNode*) malloc(sizeof(struct FunctionNode));
            if (fnew == NULL) {
                mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
                return (ERR_NO_MEMORY);
            }
            fnew->next = m_functions;
            m_functions = fnew;
            fnew->full_name = static_cast<char*>(malloc(strlen(lastline) + 1));
            if (fnew->full_name == NULL) {
                mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
                return (ERR_NO_MEMORY);
            }
            strcpy(fnew->full_name, lastline);
            fnew->short_name = static_cast<char*>(malloc(strlen(lastline) + 1));
            if (fnew->short_name == NULL) return (ERR_NO_MEMORY);
            strcpy(fnew->short_name, "");
            char copyline[MAX_LINE_LENGTH], * p1, * p2;
            strcpy(copyline, lastline);
            p1 = strchr(copyline, '(');
            if (p1) {
                p2 = p1;
                *p1 = 0;
                while (p2 != copyline && *p2 != ' ')p2--;
                if (*p2 == ' ') strcpy(fnew->short_name, p2 + 1);
            }

            fnew->ins_array = static_cast<struct Instructions*>(malloc(sizeof(struct Instruction)));
            if (fnew->ins_array == NULL) {
                mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
                return (ERR_NO_MEMORY);
            }
            fnew->ins_array->array = static_cast<struct Instruction**>(malloc(
                    ALLOC_STEP * sizeof(struct Instruction*)));
            fnew->ins_array->filled_elements = 0;
            fnew->ins_array->maximum_elements = ALLOC_STEP;

            m_numFunctions++;
            continue;
        }
        // instructions
        if (m_functions == NULL)
            return (ERR_DATA_MISMATCH);
        if (m_functions->ins_array->filled_elements >= m_functions->ins_array->maximum_elements) {
            m_functions->ins_array->array = static_cast<struct Instruction**>(realloc(m_functions->ins_array->array,
                                                                                      (m_functions->ins_array->maximum_elements +
                                                                                       ALLOC_STEP) *
                                                                                      sizeof(struct Instruction*)));
            if (m_functions->ins_array->array == NULL) {
                mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
                return (ERR_NO_MEMORY);
            }
            m_functions->ins_array->maximum_elements += ALLOC_STEP;
        }
        m_functions->ins_array->array[m_functions->ins_array->filled_elements] = static_cast<struct Instruction*>(malloc(
                sizeof(struct Instruction)));
        if (m_functions->ins_array->array[m_functions->ins_array->filled_elements] == NULL) {
            mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
            return (ERR_NO_MEMORY);
        }
        m_functions->ins_array->array[m_functions->ins_array->filled_elements]->full_line = static_cast<char*>(malloc(
                strlen(line) + 1));
        if (m_functions->ins_array->array[m_functions->ins_array->filled_elements]->full_line == NULL) {
            mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
            return (ERR_NO_MEMORY);
        }
        strcpy(m_functions->ins_array->array[m_functions->ins_array->filled_elements]->full_line, line);
        m_functions->ins_array->array[m_functions->ins_array->filled_elements]->line_number = atoi(line);
        m_functions->ins_array->array[m_functions->ins_array->filled_elements]->instruction = static_cast<char*>(malloc(
                strlen(line) + 1));
        if (m_functions->ins_array->array[m_functions->ins_array->filled_elements]->instruction == NULL) {
            mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
            return (ERR_NO_MEMORY);
        }
        char copyline[MAX_LINE_LENGTH], * p1, * p2;
        strcpy(copyline, line);
        p1 = strchr(copyline, ':');
        if (!p1)
            return (ERR_DATA_MISMATCH);
        p1++;
        while (white(*p1))p1++;
        p2 = p1;
        while (*p2 && !white(*p2))p2++;
        char s = *p2;
        *p2 = 0;
        strcpy(m_functions->ins_array->array[m_functions->ins_array->filled_elements]->instruction, p1);
        m_functions->ins_array->array[m_functions->ins_array->filled_elements]->instruction_code = code(
                m_functions->ins_array->array[m_functions->ins_array->filled_elements]->instruction);
        m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param1 = static_cast<char*>(malloc(
                strlen(line) + 1));
        if (m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param1 == NULL) {
            mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
            return (ERR_NO_MEMORY);
        }
        strcpy(m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param1, "");
        m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param2 = static_cast<char*>(malloc(
                strlen(line) + 1));
        if (m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param2 == NULL) {
            mainLogger.out(LOGGER_ERROR) << "Cannot allocate memory." << endl;
            return (ERR_NO_MEMORY);
        }
        strcpy(m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param2, "");
        *p2 = s;
        if (*p2) {
            while (*p2 && white(*p2))p2++;
            if (*p2) {
                char* p3 = p2;
                while (*p3 && (!white(*p3) && *p3 != ',' && *p3 != ';'))p3++;
                s = *p3;
                *p3 = 0;
                strcpy(m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param1, p2);
                *p3 = s;
                if (*p3 == ',') {
                    p3++;
                    while (*p3 && white(*p3))p3++;
                    if (*p3) {
                        char* p4 = p3;
                        while (*p4 && (!white(*p4) && *p4 != ',' && *p4 != ';'))p4++;
                        s = *p4;
                        *p4 = 0;
                        strcpy(m_functions->ins_array->array[m_functions->ins_array->filled_elements]->param2, p3);
                    }
                }
            }
        }
        m_functions->ins_array->filled_elements++;
    }
    return ERR_NO_ERROR;
}

int JVMSimulator::jvmsim_run(int function_number, int line_from, int line_to) {
    struct FunctionNode* function = get_function_by_number(function_number);

    if (function == NULL) {
        mainLogger.out(LOGGER_ERROR) << "Cannot find function with nubmer: " << function_number << endl;
        return (ERR_NO_SUCH_FUNCTION);
    }

    destroy_state();

    struct Pc PC = {function, 0};

    PC.ln = line_from;

    int returnValue;
    int insc = EMULATE_MAX_INSTRUCTIONS;
    do {
        int result = emulate_ins(&PC, returnValue);

        if (result == END) {
            break;
        } else if (result == CONTINUE) {
            PC.ln++;
        } else if (result == GOTO_VALUE) {
            int nextIns = find_instruction_by_number(PC.currentFunction, returnValue);
            if (nextIns == -1) {
                return ERR_NO_SUCH_INSTRUCTION;
            }
            PC.ln = nextIns;
        }
        insc--;
    } while ((strcmp(PC.currentFunction->short_name, function->short_name) || PC.ln <= line_to) && insc);
    //if we are in invoked function, the line limitation is ignored.

    if (!insc)
        return (ERR_MAX_NUMBER_OF_INSTRUCTIONS_EXHAUSTED);

    return (ERR_NO_ERROR);
}

int JVMSimulator::get_num_of_functions() {
    int ret = 0;
    struct FunctionNode* current = m_functions;

    while (current != NULL) {
        ret++;
        current = current->next;
    }

    return ret;
}

struct FunctionNode* JVMSimulator::get_function_by_number(unsigned char num) {
    struct FunctionNode* current = m_functions;

    for (unsigned char i = 0; i < num; i++) {
        current = current->next;
        if (current == NULL) {
            return NULL;
        }
    }

    return current;
}

struct FunctionNode* JVMSimulator::get_function_by_name(char* name) {
    struct FunctionNode* fnct = m_functions;
    while (fnct) {
        if (!strcmp(fnct->short_name, name)) {
            return fnct;
        }
        fnct = fnct->next;
    }
    return NULL;
}

void JVMSimulator::destroy_state() {
    for (int i = 0; i < MAX_NUMBER_OF_VARIABLES; i++) {
        if (m_globalArraysUsed[i]) {
            delete m_globalArrays[i].int_array;
            m_globalArraysUsed[i] = false;
        }
        m_localsUsed[i] = false;
    }

    if (!callStackEmpty()) {
        CallStackNode* del = NULL;
        CallStackNode* current = m_callStack;
        while (current != NULL) {
            del = current;
            current = current->next;
            delete del;
        }
        m_callStack = NULL;
    }
}

int JVMSimulator::find_instruction_by_number(FunctionNode* function, int instructionNumber) {
    for (int i = 0; i < function->ins_array->filled_elements; i++) {
        if (function->ins_array->array[i]->line_number == instructionNumber) {
            return i;
        }
    }
    return -1;
}

int JVMSimulator::get_next_global_index() {
    for (int i = 0; i < MAX_NUMBER_OF_VARIABLES; i++) {
        if (!m_globalArraysUsed[i]) {
            return i;
        }
    }
    return -1;
}

bool JVMSimulator::hasFourIntegersOnStack() {
    if (m_stack == NULL || m_stack->next == NULL || m_stack->next->next == NULL || m_stack->next->next->next == NULL)
        return false;
    return m_stack->data_type == STACKTYPE_INTEGER && m_stack->next->data_type == STACKTYPE_INTEGER &&
           m_stack->next->next->data_type == STACKTYPE_INTEGER &&
           m_stack->next->next->next->data_type == STACKTYPE_INTEGER;
}

bool JVMSimulator::hasThreeIntegersOnStack() {
    if (m_stack == NULL || m_stack->next == NULL || m_stack->next->next == NULL) return false;
    return m_stack->data_type == STACKTYPE_INTEGER && m_stack->next->data_type == STACKTYPE_INTEGER && m_stack->next->next->data_type == STACKTYPE_INTEGER;
}

bool JVMSimulator::hasTwoIntegersOnStack() {
    if (m_stack == NULL || m_stack->next == NULL) return false;
    return m_stack->data_type == STACKTYPE_INTEGER && m_stack->next->data_type == STACKTYPE_INTEGER;
}

bool JVMSimulator::hasIntegerOnStack() {
    return m_stack != NULL && m_stack->data_type == STACKTYPE_INTEGER;
}

bool JVMSimulator::hasArrayRefOnStack() {
    return m_stack != NULL && m_stack->data_type == STACKTYPE_ARRAYREF;
}

bool JVMSimulator::hasTwoArrRefOnStack() {
    if (m_stack == NULL || m_stack->next == NULL) return false;
    return m_stack->data_type == STACKTYPE_ARRAYREF && m_stack->next->data_type == STACKTYPE_ARRAYREF;
}

int64_t JVMSimulator::createLongFromTwoInts(int32_t first, int32_t second) {
    int64_t ret = first;
    ret = (ret << 32) | second;
    return ret;
}
