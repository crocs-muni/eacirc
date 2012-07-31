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

    unsigned char VAR_1_4_DIV = VAR_IN_4 /  ((VAR_IN_12 != 0) ? VAR_IN_12 : 1) / ((VAR_IN_15 != 0) ? VAR_IN_15 : 1) / 1;
    unsigned char VAR_1_6_NOP = VAR_IN_6 ;
    unsigned char VAR_1_7_ROL_7 = VAR_IN_7 << 7 ;
    unsigned char VAR_2_3_NOR = 0 | ~ VAR_1_4_DIV | ~ 0xff;
    unsigned char VAR_2_4_NOP = VAR_1_4_DIV ;
    unsigned char VAR_2_5_AND = VAR_1_6_NOP & VAR_1_7_ROL_7 & 0xff;
    unsigned char VAR_3_3_NOP = VAR_2_3_NOR ;
    unsigned char VAR_3_4_SUB = VAR_2_4_NOP -  VAR_2_5_AND - 0;
    unsigned char VAR_4_3_SUB = VAR_3_3_NOP -  VAR_3_4_SUB - 0;
    unsigned char VAR_5_0_SUM = VAR_4_3_SUB + 0;
    unsigned char VAR_5_1_SUM = VAR_4_3_SUB + 0;

    outputs[0] = VAR_5_0_SUM;
    outputs[1] = VAR_5_1_SUM;

}