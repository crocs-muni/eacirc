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

    unsigned char VAR_1_3_NOP = VAR_IN_3 ;
    unsigned char VAR_1_5_OR_ = VAR_IN_3 | VAR_IN_6 | VAR_IN_11 | VAR_IN_12 | VAR_IN_13 | VAR_IN_14 | 0;
    unsigned char VAR_2_3_BSL_0 = VAR_1_3_NOP & 0 ;
    unsigned char VAR_2_5_ROL_2 = VAR_1_5_OR_ << 2 ;
    unsigned char VAR_3_3_ADD = VAR_2_3_BSL_0 +  VAR_2_5_ROL_2 + 0;
    unsigned char VAR_4_3_NOP = VAR_3_3_ADD ;
    unsigned char VAR_5_0_SUM = VAR_4_3_NOP + 0;
    unsigned char VAR_5_1_XOR = VAR_4_3_NOP ^ 0;

    outputs[0] = VAR_5_0_SUM;
    outputs[1] = VAR_5_1_XOR;

}