\ Model WindFarmTwoPrice_Hour
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  128.2 p_DA[0] + 120.38 p_DA[1] + 117.09 p_DA[2] + 112.15 p_DA[3]
   + 109.75 p_DA[4] + 111.89 p_DA[5] + 117.11 p_DA[6] + 135.81 p_DA[7]
   + 138.82 p_DA[8] + 137.77 p_DA[9] + 125.08 p_DA[10] + 112.1 p_DA[11]
   + 102.21 p_DA[12] + 97.27 p_DA[13] + 99.39 p_DA[14] + 97.8 p_DA[15]
   + 125.41 p_DA[16] + 145 p_DA[17] + 158.55 p_DA[18] + 150.78 p_DA[19]
   + 133.41 p_DA[20] + 125.42 p_DA[21] + 122.03 p_DA[22] + 119.9 p_DA[23]
   + 108.97 pos_imbalance[0,0] + 120.38 pos_imbalance[1,0]
   + 117.09 pos_imbalance[2,0] + 112.15 pos_imbalance[3,0]
   + 93.2875 pos_imbalance[4,0] + 95.1065 pos_imbalance[5,0]
   + 99.5435 pos_imbalance[6,0] + 135.81 pos_imbalance[7,0]
   + 138.82 pos_imbalance[8,0] + 137.77 pos_imbalance[9,0]
   + 106.318 pos_imbalance[10,0] + 112.1 pos_imbalance[11,0]
   + 102.21 pos_imbalance[12,0] + 82.6795 pos_imbalance[13,0]
   + 84.4815 pos_imbalance[14,0] + 83.13 pos_imbalance[15,0]
   + 106.5985 pos_imbalance[16,0] + 145 pos_imbalance[17,0]
   + 134.7675 pos_imbalance[18,0] + 128.163 pos_imbalance[19,0]
   + 133.41 pos_imbalance[20,0] + 106.607 pos_imbalance[21,0]
   + 103.7255 pos_imbalance[22,0] + 101.915 pos_imbalance[23,0]
   - 128.2 neg_imbalance[0,0] - 150.475 neg_imbalance[1,0]
   - 146.3625 neg_imbalance[2,0] - 140.1875 neg_imbalance[3,0]
   - 109.75 neg_imbalance[4,0] - 111.89 neg_imbalance[5,0]
   - 117.11 neg_imbalance[6,0] - 169.7625 neg_imbalance[7,0]
   - 173.525 neg_imbalance[8,0] - 172.2125 neg_imbalance[9,0]
   - 125.08 neg_imbalance[10,0] - 140.125 neg_imbalance[11,0]
   - 127.7625 neg_imbalance[12,0] - 97.27 neg_imbalance[13,0]
   - 99.39 neg_imbalance[14,0] - 97.8 neg_imbalance[15,0]
   - 125.41 neg_imbalance[16,0] - 181.25 neg_imbalance[17,0]
   - 158.55 neg_imbalance[18,0] - 150.78 neg_imbalance[19,0]
   - 166.7625 neg_imbalance[20,0] - 125.42 neg_imbalance[21,0]
   - 122.03 neg_imbalance[22,0] - 119.9 neg_imbalance[23,0]
Subject To
 imbalance_0_1: - p_DA[0] - pos_imbalance[0,0] + neg_imbalance[0,0]
   = -121.6777137082392
 imbalance_1_1: - p_DA[1] - pos_imbalance[1,0] + neg_imbalance[1,0]
   = -124.0933004399001
 imbalance_2_1: - p_DA[2] - pos_imbalance[2,0] + neg_imbalance[2,0]
   = -116.0929140411366
 imbalance_3_1: - p_DA[3] - pos_imbalance[3,0] + neg_imbalance[3,0]
   = -102.319284270598
 imbalance_4_1: - p_DA[4] - pos_imbalance[4,0] + neg_imbalance[4,0]
   = -84.91112828438949
 imbalance_5_1: - p_DA[5] - pos_imbalance[5,0] + neg_imbalance[5,0]
   = -72.55528474616574
 imbalance_6_1: - p_DA[6] - pos_imbalance[6,0] + neg_imbalance[6,0]
   = -62.90512424206397
 imbalance_7_1: - p_DA[7] - pos_imbalance[7,0] + neg_imbalance[7,0]
   = -48.68802758292712
 imbalance_8_1: - p_DA[8] - pos_imbalance[8,0] + neg_imbalance[8,0]
   = -39.09032814171918
 imbalance_9_1: - p_DA[9] - pos_imbalance[9,0] + neg_imbalance[9,0]
   = -32.14807989537511
 imbalance_10_1: - p_DA[10] - pos_imbalance[10,0] + neg_imbalance[10,0]
   = -31.72705385804304
 imbalance_11_1: - p_DA[11] - pos_imbalance[11,0] + neg_imbalance[11,0]
   = -33.46614552371894
 imbalance_12_1: - p_DA[12] - pos_imbalance[12,0] + neg_imbalance[12,0]
   = -36.25847104981572
 imbalance_13_1: - p_DA[13] - pos_imbalance[13,0] + neg_imbalance[13,0]
   = -41.27853406253715
 imbalance_14_1: - p_DA[14] - pos_imbalance[14,0] + neg_imbalance[14,0]
   = -50.374063726073
 imbalance_15_1: - p_DA[15] - pos_imbalance[15,0] + neg_imbalance[15,0]
   = -62.41320889311616
 imbalance_16_1: - p_DA[16] - pos_imbalance[16,0] + neg_imbalance[16,0]
   = -91.49610628938295
 imbalance_17_1: - p_DA[17] - pos_imbalance[17,0] + neg_imbalance[17,0]
   = -123.6294733087623
 imbalance_18_1: - p_DA[18] - pos_imbalance[18,0] + neg_imbalance[18,0]
   = -155.0014861490905
 imbalance_19_1: - p_DA[19] - pos_imbalance[19,0] + neg_imbalance[19,0]
   = -171.3957912257758
 imbalance_20_1: - p_DA[20] - pos_imbalance[20,0] + neg_imbalance[20,0]
   = -182.5609321127095
 imbalance_21_1: - p_DA[21] - pos_imbalance[21,0] + neg_imbalance[21,0]
   = -214.3383664249197
 imbalance_22_1: - p_DA[22] - pos_imbalance[22,0] + neg_imbalance[22,0]
   = -220.8001426703127
 imbalance_23_1: - p_DA[23] - pos_imbalance[23,0] + neg_imbalance[23,0]
   = -217.9165675900607
Bounds
 p_DA[0] <= 500
 p_DA[1] <= 500
 p_DA[2] <= 500
 p_DA[3] <= 500
 p_DA[4] <= 500
 p_DA[5] <= 500
 p_DA[6] <= 500
 p_DA[7] <= 500
 p_DA[8] <= 500
 p_DA[9] <= 500
 p_DA[10] <= 500
 p_DA[11] <= 500
 p_DA[12] <= 500
 p_DA[13] <= 500
 p_DA[14] <= 500
 p_DA[15] <= 500
 p_DA[16] <= 500
 p_DA[17] <= 500
 p_DA[18] <= 500
 p_DA[19] <= 500
 p_DA[20] <= 500
 p_DA[21] <= 500
 p_DA[22] <= 500
 p_DA[23] <= 500
End
