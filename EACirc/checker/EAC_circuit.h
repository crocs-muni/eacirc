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

    unsigned char VAR_1_1_NOP = VAR_IN_1 ;
    unsigned char VAR_1_2_XOR = VAR_IN_2 ^ VAR_IN_5 ^ 0;
    unsigned char VAR_1_7_SUM = VAR_IN_0 + 0;
    unsigned char VAR_2_0_CONST_112 = 112 ;
    unsigned char VAR_2_1_NOP = VAR_1_1_NOP ;
    unsigned char VAR_2_2_MUL = VAR_1_2_XOR *  VAR_1_1_NOP * VAR_1_2_XOR * 1;
    unsigned char VAR_2_3_NAN = 0xff & ~  VAR_1_2_XOR & ~ 0;
    unsigned char VAR_2_5_OR_ = VAR_1_7_SUM | 0;
    unsigned char VAR_3_1_ADD = VAR_2_1_NOP +  VAR_2_0_CONST_112 + VAR_2_1_NOP + VAR_2_2_MUL + VAR_2_3_NAN + 0;
    unsigned char VAR_3_3_OR_ = VAR_2_2_MUL | VAR_2_5_OR_ | 0;
    unsigned char VAR_4_1_XOR = VAR_3_1_ADD ^ VAR_3_3_OR_ ^ 0;
    unsigned char VAR_5_0_OR_ = VAR_4_1_XOR | 0;
    unsigned char VAR_5_1_NOP = VAR_4_1_XOR ;

    outputs[0] = VAR_5_0_OR_;
    outputs[1] = VAR_5_1_NOP;

}