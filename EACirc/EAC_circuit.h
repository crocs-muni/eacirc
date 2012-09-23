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

    unsigned char VAR_1_1_NOP = VAR_IN_1 ;
    unsigned char VAR_1_2_NOP = VAR_IN_2 ;
    unsigned char VAR_1_3_MUL = VAR_IN_3 *  VAR_IN_3 * VAR_IN_13 * 1;
    unsigned char VAR_1_4_OR_ = VAR_IN_14 | 0;
    unsigned char VAR_2_0_NAN = 0xff & ~  VAR_1_2_NOP & ~ 0;
    unsigned char VAR_2_2_NOR = 0 | ~ VAR_1_1_NOP | ~ VAR_1_2_NOP | ~ VAR_1_3_MUL | ~ 0xff;
    unsigned char VAR_2_3_CONST_145 = 145 ;
    unsigned char VAR_2_5_OR_ = VAR_1_4_OR_ | 0;
    unsigned char VAR_3_0_ROL_5 = VAR_2_0_NAN << 5 ;
    unsigned char VAR_3_1_AND = VAR_2_0_NAN & VAR_2_3_CONST_145 & 0xff;
    unsigned char VAR_3_2_OR_ = VAR_2_2_NOR | VAR_2_3_CONST_145 | 0;
    unsigned char VAR_3_3_SUB = VAR_2_3_CONST_145 -  VAR_2_2_NOR - VAR_2_3_CONST_145 - 0;
    unsigned char VAR_3_5_NOP = VAR_2_5_OR_ ;
    unsigned char VAR_4_0_NOP = VAR_3_0_ROL_5 ;
    unsigned char VAR_4_2_MUL = VAR_3_2_OR_ *  VAR_3_1_AND * 1;
    unsigned char VAR_4_3_NOR = 0 | ~ VAR_3_3_SUB | ~ VAR_3_5_NOP | ~ 0xff;
    unsigned char VAR_5_0_OR_ = VAR_4_0_NOP | VAR_4_2_MUL | VAR_4_3_NOR | 0;
    unsigned char VAR_5_1_OR_ = VAR_4_0_NOP | VAR_4_2_MUL | VAR_4_3_NOR | 0;

    outputs[0] = VAR_5_0_OR_;
    outputs[1] = VAR_5_1_OR_;

}