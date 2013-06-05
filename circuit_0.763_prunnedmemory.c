int headerCircuit_inputLayerSize = 19;
int headerCircuit_outputLayerSize = 5;

static void circuit(unsigned char inputs[64], unsigned char outputs[2]) {
    const int SECTOR_SIZE = 16;
    const int NUM_SECTORS = 4;

    unsigned char VAR_IN_0 = 0;
    unsigned char VAR_IN_1 = 0;
    unsigned char VAR_IN_2 = 0;
    unsigned char VAR_IN_3 = 0;
    unsigned char VAR_IN_4 = 0;
    unsigned char VAR_IN_5 = 0;
    unsigned char VAR_IN_6 = 0;
    unsigned char VAR_IN_7 = 0;
    unsigned char VAR_IN_8 = 0;
    unsigned char VAR_IN_9 = 0;
    unsigned char VAR_IN_10 = 0;
    unsigned char VAR_IN_11 = 0;
    unsigned char VAR_IN_12 = 0;
    unsigned char VAR_IN_13 = 0;
    unsigned char VAR_IN_14 = 0;
    unsigned char VAR_IN_15 = 0;
    unsigned char VAR_IN_16 = 0;
    unsigned char VAR_IN_17 = 0;
    unsigned char VAR_IN_18 = 0;

    for (int sector = 0; sector < NUM_SECTORS; sector++) {
        VAR_IN_3 = inputs[sector * SECTOR_SIZE + 0];
        VAR_IN_4 = inputs[sector * SECTOR_SIZE + 1];
        VAR_IN_5 = inputs[sector * SECTOR_SIZE + 2];
        VAR_IN_6 = inputs[sector * SECTOR_SIZE + 3];
        VAR_IN_7 = inputs[sector * SECTOR_SIZE + 4];
        VAR_IN_8 = inputs[sector * SECTOR_SIZE + 5];
        VAR_IN_9 = inputs[sector * SECTOR_SIZE + 6];
        VAR_IN_10 = inputs[sector * SECTOR_SIZE + 7];
        VAR_IN_11 = inputs[sector * SECTOR_SIZE + 8];
        VAR_IN_12 = inputs[sector * SECTOR_SIZE + 9];
        VAR_IN_13 = inputs[sector * SECTOR_SIZE + 10];
        VAR_IN_14 = inputs[sector * SECTOR_SIZE + 11];
        VAR_IN_15 = inputs[sector * SECTOR_SIZE + 12];
        VAR_IN_16 = inputs[sector * SECTOR_SIZE + 13];
        VAR_IN_17 = inputs[sector * SECTOR_SIZE + 14];
        VAR_IN_18 = inputs[sector * SECTOR_SIZE + 15];

        unsigned char VAR_1_14_XOR = VAR_IN_14 ^ 0;
        unsigned char VAR_1_16_XOR = VAR_IN_16 ^ 0;
        unsigned char VAR_2_15_SUM = VAR_1_14_XOR + VAR_1_16_XOR + 0;
        unsigned char VAR_2_16_NOP = VAR_1_16_XOR  ;
        unsigned char VAR_2_17_SUM = VAR_1_16_XOR + 0;
        unsigned char VAR_3_14_AND = VAR_2_15_SUM & 0xff;
        unsigned char VAR_3_16_AND = VAR_2_15_SUM & VAR_2_17_SUM & 0xff;
        unsigned char VAR_3_17_MUL = VAR_2_16_NOP * 1;
        unsigned char VAR_4_15_NOP = VAR_3_14_AND  ;
        unsigned char VAR_4_16_SUB = VAR_3_16_AND - 0;
        unsigned char VAR_4_17_ROL_0 = VAR_3_17_MUL << 0 ;
        unsigned char VAR_5_0_NOP = 0;
        unsigned char VAR_5_1_NOP = 0;
        unsigned char VAR_5_2_NOP = 0;
        unsigned char VAR_5_3_SUM = VAR_4_15_NOP + VAR_4_16_SUB + VAR_4_17_ROL_0 + 0;
        unsigned char VAR_5_4_ADD = VAR_4_17_ROL_0 + 0;

        VAR_IN_0 = VAR_5_0_NOP;
        VAR_IN_1 = VAR_5_1_NOP;
        VAR_IN_2 = VAR_5_2_NOP;
        outputs[0] = VAR_5_3_SUM;
        outputs[1] = VAR_5_4_ADD;
    }

}