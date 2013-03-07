static void circuit(unsigned char inputs[MAX_INPUTS], unsigned char outputs[MAX_OUTPUTS]) {
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

    unsigned char VAR_1_2_SUB = VAR_IN_2 -  VAR_IN_2 - VAR_IN_14 - 0;
    unsigned char VAR_1_5_SUB = VAR_IN_5 -  VAR_IN_5 - VAR_IN_12 - 0;
    unsigned char VAR_2_1_CONST_8 = 8 ;
    unsigned char VAR_2_3_NAN = 0xff & ~  VAR_1_2_SUB & ~ VAR_1_5_SUB & ~ 0;
    unsigned char VAR_3_2_SUM = VAR_2_1_CONST_8 + VAR_2_3_NAN + 0;
    unsigned char VAR_3_3_NAN = 0xff & ~  VAR_2_3_NAN & ~ 0;
    unsigned char VAR_4_3_NAN = 0xff & ~  VAR_3_2_SUM & ~ VAR_3_3_NAN & ~ 0;
    unsigned char VAR_5_0_NOR = 0 | ~ VAR_4_3_NAN | ~ 0xff;
    unsigned char VAR_5_1_NAN = 0xff & ~  VAR_4_3_NAN & ~ 0;

    outputs[0] = VAR_5_0_NOR;
    outputs[1] = VAR_5_1_NAN;

}