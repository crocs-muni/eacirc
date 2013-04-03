int headerCircuit_inputLayerSize = 16;
int headerCircuit_outputLayerSize = 2;

static void circuit(unsigned char inputs[16], unsigned char outputs[2]) {
    unsigned char VAR_IN_0 = inputs[0];
    unsigned char VAR_IN_1 = inputs[1];
    unsigned char VAR_IN_2 = inputs[2];
    unsigned char VAR_IN_3 = inputs[3];
    unsigned char VAR_IN_4 = inputs[4];
    unsigned char VAR_IN_5 = inputs[5];
    unsigned char VAR_IN_6 = inputs[6];
    unsigned char VAR_IN_7 = inputs[7];
    unsigned char VAR_IN_8 = inputs[8];
    unsigned char VAR_IN_9 = inputs[9];
    unsigned char VAR_IN_10 = inputs[10];
    unsigned char VAR_IN_11 = inputs[11];
    unsigned char VAR_IN_12 = inputs[12];
    unsigned char VAR_IN_13 = inputs[13];
    unsigned char VAR_IN_14 = inputs[14];
    unsigned char VAR_IN_15 = inputs[15];

    unsigned char VAR_1_0_NOP = VAR_IN_0 ;
    unsigned char VAR_1_2_NOP = VAR_IN_2 ;
    unsigned char VAR_1_3_BSL_0 = VAR_IN_3 & 0 ;
    unsigned char VAR_1_4_NOP = VAR_IN_4 ;
    unsigned char VAR_2_0_NOP = VAR_1_0_NOP ;
    unsigned char VAR_2_2_DIV = VAR_1_2_NOP /  ((VAR_1_2_NOP != 0) ? VAR_1_2_NOP : 1) / 1;
    unsigned char VAR_2_3_NOP = VAR_1_3_BSL_0 ;
    unsigned char VAR_2_4_ROR_1 = VAR_1_4_NOP >> 1 ;
    unsigned char VAR_3_0_MUL = VAR_2_0_NOP *  VAR_2_3_NOP * 1;
    unsigned char VAR_3_3_OR_ = VAR_2_2_DIV | VAR_2_4_ROR_1 | 0;
    unsigned char VAR_4_0_SUB = VAR_3_0_MUL -  VAR_3_3_OR_ - 0;
    unsigned char VAR_5_0_NOP = VAR_4_0_SUB ;
    unsigned char VAR_5_1_OR_ = VAR_4_0_SUB | 0;

    outputs[0] = VAR_5_0_NOP;
    outputs[1] = VAR_5_1_OR_;

}