\ Model WindFarmTwoPrice_Hour
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  64.26 p_DA[0] + 63.608 p_DA[1] + 69.168 p_DA[2] + 64.568 p_DA[3]
   + 66.432 p_DA[4] + 68.178 p_DA[5] + 85.224 p_DA[6] + 96.226 p_DA[7]
   + 86.674 p_DA[8] + 70.552 p_DA[9] + 48.668 p_DA[10] + 33.194 p_DA[11]
   + 28.406 p_DA[12] + 27.222 p_DA[13] + 35.55 p_DA[14] + 49.89 p_DA[15]
   + 75.23 p_DA[16] + 104.374 p_DA[17] + 119.658 p_DA[18]
   + 111.364 p_DA[19] + 92.914 p_DA[20] + 79.622 p_DA[21] + 73.604 p_DA[22]
   + 67.192 p_DA[23] + 0 pos_imbalance[0,0] + 18.4433 pos_imbalance[0,1]
   + 5.2122 pos_imbalance[0,2] + 15.6706 pos_imbalance[0,3]
   + 15.2949 pos_imbalance[0,4] + 0 pos_imbalance[1,0]
   + 20.444 pos_imbalance[1,1] + 8.022 pos_imbalance[1,2]
   + 17.992 pos_imbalance[1,3] + 17.15 pos_imbalance[1,4]
   + 0 pos_imbalance[2,0] + 19.792 pos_imbalance[2,1]
   + 13.192 pos_imbalance[2,2] + 15.2507 pos_imbalance[2,3]
   + 15.914 pos_imbalance[2,4] + 0 pos_imbalance[3,0]
   + 19.338 pos_imbalance[3,1] + 16.18 pos_imbalance[3,2]
   + 13.42 pos_imbalance[3,3] + 15.63 pos_imbalance[3,4]
   + 0 pos_imbalance[4,0] + 16.7161 pos_imbalance[4,1]
   + 17.168 pos_imbalance[4,2] + 13.888 pos_imbalance[4,3]
   + 15.71 pos_imbalance[4,4] + 0 pos_imbalance[5,0]
   + 17.9979 pos_imbalance[5,1] + 17.5661 pos_imbalance[5,2]
   + 17.3876 pos_imbalance[5,3] + 4.9997 pos_imbalance[5,4]
   + 0 pos_imbalance[6,0] + 22.559 pos_imbalance[6,1]
   + 27.052 pos_imbalance[6,2] + 25.604 pos_imbalance[6,3]
   + 5.1238 pos_imbalance[6,4] + 0 pos_imbalance[7,0]
   + 32.148 pos_imbalance[7,1] + 26.0032 pos_imbalance[7,2]
   + 23.2441 pos_imbalance[7,3] + 5.219 pos_imbalance[7,4]
   + 0 pos_imbalance[8,0] + 32.198 pos_imbalance[8,1]
   + 23.0044 pos_imbalance[8,2] + 18.1152 pos_imbalance[8,3]
   + 6.1 pos_imbalance[8,4] + 0 pos_imbalance[9,0]
   + 30.96 pos_imbalance[9,1] + 20.154 pos_imbalance[9,2]
   + 13.692 pos_imbalance[9,3] + 5.746 pos_imbalance[9,4]
   + 0 pos_imbalance[10,0] + 21.0528 pos_imbalance[10,1]
   + 13.64 pos_imbalance[10,2] + 5.448 pos_imbalance[10,3]
   + 4.0902 pos_imbalance[10,4] + 0 pos_imbalance[11,0]
   + 22.15 pos_imbalance[11,1] + 10.106 pos_imbalance[11,2]
   + 0.358 pos_imbalance[11,3] + 0.493 pos_imbalance[11,4]
   + 0 pos_imbalance[12,0] + 22.016 pos_imbalance[12,1]
   + 5.2938 pos_imbalance[12,2] - 0.034 pos_imbalance[12,3]
   + 0.1717 pos_imbalance[12,4] + 0 pos_imbalance[13,0]
   + 18.4076 pos_imbalance[13,1] + 4.7464 pos_imbalance[13,2]
   - 0.0476 pos_imbalance[13,3] + 0.0323 pos_imbalance[13,4]
   + 0 pos_imbalance[14,0] + 18.292 pos_imbalance[14,1]
   + 10.024 pos_imbalance[14,2] + 3.472 pos_imbalance[14,3]
   + 0.534 pos_imbalance[14,4] + 0 pos_imbalance[15,0]
   + 18.7034 pos_imbalance[15,1] + 11.9034 pos_imbalance[15,2]
   + 6.562 pos_imbalance[15,3] + 6.162 pos_imbalance[15,4]
   + 0 pos_imbalance[16,0] + 20.6652 pos_imbalance[16,1]
   + 16.7603 pos_imbalance[16,2] + 16.3251 pos_imbalance[16,3]
   + 11.994 pos_imbalance[16,4] + 0 pos_imbalance[17,0]
   + 28.59 pos_imbalance[17,1] + 24.3338 pos_imbalance[17,2]
   + 23.987 pos_imbalance[17,3] + 16.0956 pos_imbalance[17,4]
   + 0 pos_imbalance[18,0] + 28.2353 pos_imbalance[18,1]
   + 30.1614 pos_imbalance[18,2] + 29.0989 pos_imbalance[18,3]
   + 14.2137 pos_imbalance[18,4] + 0 pos_imbalance[19,0]
   + 29.4814 pos_imbalance[19,1] + 34.018 pos_imbalance[19,2]
   + 33.554 pos_imbalance[19,3] + 9.108 pos_imbalance[19,4]
   + 0 pos_imbalance[20,0] + 29.798 pos_imbalance[20,1]
   + 24.0686 pos_imbalance[20,2] + 24.2267 pos_imbalance[20,3]
   + 6.298 pos_imbalance[20,4] + 0 pos_imbalance[21,0]
   + 23.0673 pos_imbalance[21,1] + 22.71 pos_imbalance[21,2]
   + 23.554 pos_imbalance[21,3] + 6.22 pos_imbalance[21,4]
   + 0 pos_imbalance[22,0] + 20.9797 pos_imbalance[22,1]
   + 18.0982 pos_imbalance[22,2] + 18.2682 pos_imbalance[22,3]
   + 6.138 pos_imbalance[22,4] + 0 pos_imbalance[23,0]
   + 18.9601 pos_imbalance[23,1] + 18.698 pos_imbalance[23,2]
   + 20.1 pos_imbalance[23,3] + 5.1748 pos_imbalance[23,4]
   + 0 neg_imbalance[0,0] - 21.698 neg_imbalance[0,1]
   - 6.132 neg_imbalance[0,2] - 18.436 neg_imbalance[0,3]
   - 17.994 neg_imbalance[0,4] + 0 neg_imbalance[1,0]
   - 25.555 neg_imbalance[1,1] - 10.0275 neg_imbalance[1,2]
   - 22.49 neg_imbalance[1,3] - 21.4375 neg_imbalance[1,4]
   + 0 neg_imbalance[2,0] - 24.74 neg_imbalance[2,1]
   - 15.52 neg_imbalance[2,2] - 17.942 neg_imbalance[2,3]
   - 19.8925 neg_imbalance[2,4] + 0 neg_imbalance[3,0]
   - 24.1725 neg_imbalance[3,1] - 20.225 neg_imbalance[3,2]
   - 16.775 neg_imbalance[3,3] - 19.5375 neg_imbalance[3,4]
   + 0 neg_imbalance[4,0] - 19.666 neg_imbalance[4,1]
   - 21.46 neg_imbalance[4,2] - 17.36 neg_imbalance[4,3]
   - 19.6375 neg_imbalance[4,4] + 0 neg_imbalance[5,0]
   - 21.174 neg_imbalance[5,1] - 20.666 neg_imbalance[5,2]
   - 20.456 neg_imbalance[5,3] - 5.882 neg_imbalance[5,4]
   + 0 neg_imbalance[6,0] - 26.54 neg_imbalance[6,1]
   - 33.815 neg_imbalance[6,2] - 32.005 neg_imbalance[6,3]
   - 6.028 neg_imbalance[6,4] + 0 neg_imbalance[7,0]
   - 40.185 neg_imbalance[7,1] - 30.592 neg_imbalance[7,2]
   - 27.346 neg_imbalance[7,3] - 6.14 neg_imbalance[7,4]
   + 0 neg_imbalance[8,0] - 40.2475 neg_imbalance[8,1]
   - 27.064 neg_imbalance[8,2] - 21.312 neg_imbalance[8,3]
   - 7.625 neg_imbalance[8,4] + 0 neg_imbalance[9,0]
   - 38.7 neg_imbalance[9,1] - 25.1925 neg_imbalance[9,2]
   - 17.115 neg_imbalance[9,3] - 7.1825 neg_imbalance[9,4]
   + 0 neg_imbalance[10,0] - 24.768 neg_imbalance[10,1]
   - 17.05 neg_imbalance[10,2] - 6.81 neg_imbalance[10,3]
   - 4.812 neg_imbalance[10,4] + 0 neg_imbalance[11,0]
   - 27.6875 neg_imbalance[11,1] - 12.6325 neg_imbalance[11,2]
   - 0.4475 neg_imbalance[11,3] - 0.58 neg_imbalance[11,4]
   + 0 neg_imbalance[12,0] - 27.52 neg_imbalance[12,1]
   - 6.228 neg_imbalance[12,2] + 0.04 neg_imbalance[12,3]
   - 0.202 neg_imbalance[12,4] + 0 neg_imbalance[13,0]
   - 21.656 neg_imbalance[13,1] - 5.584 neg_imbalance[13,2]
   + 0.056 neg_imbalance[13,3] - 0.038 neg_imbalance[13,4]
   + 0 neg_imbalance[14,0] - 21.52 neg_imbalance[14,1]
   - 12.53 neg_imbalance[14,2] - 4.34 neg_imbalance[14,3]
   - 0.6675 neg_imbalance[14,4] + 0 neg_imbalance[15,0]
   - 22.004 neg_imbalance[15,1] - 14.004 neg_imbalance[15,2]
   - 7.72 neg_imbalance[15,3] - 7.7025 neg_imbalance[15,4]
   + 0 neg_imbalance[16,0] - 24.312 neg_imbalance[16,1]
   - 19.718 neg_imbalance[16,2] - 19.206 neg_imbalance[16,3]
   - 14.9925 neg_imbalance[16,4] + 0 neg_imbalance[17,0]
   - 35.7375 neg_imbalance[17,1] - 28.628 neg_imbalance[17,2]
   - 28.22 neg_imbalance[17,3] - 18.936 neg_imbalance[17,4]
   + 0 neg_imbalance[18,0] - 33.218 neg_imbalance[18,1]
   - 35.484 neg_imbalance[18,2] - 34.234 neg_imbalance[18,3]
   - 16.722 neg_imbalance[18,4] + 0 neg_imbalance[19,0]
   - 34.684 neg_imbalance[19,1] - 42.5225 neg_imbalance[19,2]
   - 41.9425 neg_imbalance[19,3] - 11.385 neg_imbalance[19,4]
   + 0 neg_imbalance[20,0] - 37.2475 neg_imbalance[20,1]
   - 28.316 neg_imbalance[20,2] - 28.502 neg_imbalance[20,3]
   - 7.8725 neg_imbalance[20,4] + 0 neg_imbalance[21,0]
   - 27.138 neg_imbalance[21,1] - 28.3875 neg_imbalance[21,2]
   - 29.4425 neg_imbalance[21,3] - 7.775 neg_imbalance[21,4]
   + 0 neg_imbalance[22,0] - 24.682 neg_imbalance[22,1]
   - 21.292 neg_imbalance[22,2] - 21.492 neg_imbalance[22,3]
   - 7.6725 neg_imbalance[22,4] + 0 neg_imbalance[23,0]
   - 22.306 neg_imbalance[23,1] - 23.3725 neg_imbalance[23,2]
   - 25.125 neg_imbalance[23,3] - 6.088 neg_imbalance[23,4]
Subject To
 imbalance_0_1: - p_DA[0] - pos_imbalance[0,1] + neg_imbalance[0,1]
   = -122.156996789918
 imbalance_0_2: - p_DA[0] - pos_imbalance[0,2] + neg_imbalance[0,2]
   = -121.6777137082392
 imbalance_0_3: - p_DA[0] - pos_imbalance[0,3] + neg_imbalance[0,3]
   = -45.86880275829272
 imbalance_0_4: - p_DA[0] - pos_imbalance[0,4] + neg_imbalance[0,4]
   = -283.5085007727976
 imbalance_1_1: - p_DA[1] - pos_imbalance[1,1] + neg_imbalance[1,1]
   = -126.8584294376412
 imbalance_1_2: - p_DA[1] - pos_imbalance[1,2] + neg_imbalance[1,2]
   = -124.0933004399001
 imbalance_1_3: - p_DA[1] - pos_imbalance[1,3] + neg_imbalance[1,3]
   = -48.41115800737129
 imbalance_1_4: - p_DA[1] - pos_imbalance[1,4] + neg_imbalance[1,4]
   = -287.5214005469028
 imbalance_2_1: - p_DA[2] - pos_imbalance[2,1] + neg_imbalance[2,1]
   = -132.5052015218167
 imbalance_2_2: - p_DA[2] - pos_imbalance[2,2] + neg_imbalance[2,2]
   = -116.0929140411366
 imbalance_2_3: - p_DA[2] - pos_imbalance[2,3] + neg_imbalance[2,3]
   = -46.14968493639282
 imbalance_2_4: - p_DA[2] - pos_imbalance[2,4] + neg_imbalance[2,4]
   = -295.1936452264891
 imbalance_3_1: - p_DA[3] - pos_imbalance[3,1] + neg_imbalance[3,1]
   = -133.2784448935917
 imbalance_3_2: - p_DA[3] - pos_imbalance[3,2] + neg_imbalance[3,2]
   = -102.319284270598
 imbalance_3_3: - p_DA[3] - pos_imbalance[3,3] + neg_imbalance[3,3]
   = -46.64100582570443
 imbalance_3_4: - p_DA[3] - pos_imbalance[3,4] + neg_imbalance[3,4]
   = -290.8870823921056
 imbalance_4_1: - p_DA[4] - pos_imbalance[4,1] + neg_imbalance[4,1]
   = -141.4956604446558
 imbalance_4_2: - p_DA[4] - pos_imbalance[4,2] + neg_imbalance[4,2]
   = -84.91112828438949
 imbalance_4_3: - p_DA[4] - pos_imbalance[4,3] + neg_imbalance[4,3]
   = -62.52793960290097
 imbalance_4_4: - p_DA[4] - pos_imbalance[4,4] + neg_imbalance[4,4]
   = -309.738289145167
 imbalance_5_1: - p_DA[5] - pos_imbalance[5,1] + neg_imbalance[5,1]
   = -148.4374628462728
 imbalance_5_2: - p_DA[5] - pos_imbalance[5,2] + neg_imbalance[5,2]
   = -72.55528474616574
 imbalance_5_3: - p_DA[5] - pos_imbalance[5,3] + neg_imbalance[5,3]
   = -68.8073653548924
 imbalance_5_4: - p_DA[5] - pos_imbalance[5,4] + neg_imbalance[5,4]
   = -297.0250267506837
 imbalance_6_1: - p_DA[6] - pos_imbalance[6,1] + neg_imbalance[6,1]
   = -145.3282903340863
 imbalance_6_2: - p_DA[6] - pos_imbalance[6,2] + neg_imbalance[6,2]
   = -62.90512424206397
 imbalance_6_3: - p_DA[6] - pos_imbalance[6,3] + neg_imbalance[6,3]
   = -65.17105576031388
 imbalance_6_4: - p_DA[6] - pos_imbalance[6,4] + neg_imbalance[6,4]
   = -269.1627036024254
 imbalance_7_1: - p_DA[7] - pos_imbalance[7,1] + neg_imbalance[7,1]
   = -118.3279336583046
 imbalance_7_2: - p_DA[7] - pos_imbalance[7,2] + neg_imbalance[7,2]
   = -48.68802758292712
 imbalance_7_3: - p_DA[7] - pos_imbalance[7,3] + neg_imbalance[7,3]
   = -71.38211865414338
 imbalance_7_4: - p_DA[7] - pos_imbalance[7,4] + neg_imbalance[7,4]
   = -227.4574961360123
 imbalance_8_1: - p_DA[8] - pos_imbalance[8,1] + neg_imbalance[8,1]
   = -73.53079300915468
 imbalance_8_2: - p_DA[8] - pos_imbalance[8,2] + neg_imbalance[8,2]
   = -39.09032814171918
 imbalance_8_3: - p_DA[8] - pos_imbalance[8,3] + neg_imbalance[8,3]
   = -68.16728094162407
 imbalance_8_4: - p_DA[8] - pos_imbalance[8,4] + neg_imbalance[8,4]
   = -190.2459576744739
 imbalance_9_1: - p_DA[9] - pos_imbalance[9,1] + neg_imbalance[9,1]
   = -54.17875401260255
 imbalance_9_2: - p_DA[9] - pos_imbalance[9,2] + neg_imbalance[9,2]
   = -32.14807989537511
 imbalance_9_3: - p_DA[9] - pos_imbalance[9,3] + neg_imbalance[9,3]
   = -66.8481750089169
 imbalance_9_4: - p_DA[9] - pos_imbalance[9,4] + neg_imbalance[9,4]
   = -178.3369991677565
 imbalance_10_1: - p_DA[10] - pos_imbalance[10,1] + neg_imbalance[10,1]
   = -56.00270479134466
 imbalance_10_2: - p_DA[10] - pos_imbalance[10,2] + neg_imbalance[10,2]
   = -31.72705385804304
 imbalance_10_3: - p_DA[10] - pos_imbalance[10,3] + neg_imbalance[10,3]
   = -82.42435501129474
 imbalance_10_4: - p_DA[10] - pos_imbalance[10,4] + neg_imbalance[10,4]
   = -168.1802104387112
 imbalance_11_1: - p_DA[11] - pos_imbalance[11,1] + neg_imbalance[11,1]
   = -60.02987159671859
 imbalance_11_2: - p_DA[11] - pos_imbalance[11,2] + neg_imbalance[11,2]
   = -33.46614552371894
 imbalance_11_3: - p_DA[11] - pos_imbalance[11,3] + neg_imbalance[11,3]
   = -94.07843894899537
 imbalance_11_4: - p_DA[11] - pos_imbalance[11,4] + neg_imbalance[11,4]
   = -149.6578884793722
 imbalance_12_1: - p_DA[12] - pos_imbalance[12,1] + neg_imbalance[12,1]
   = -55.4476281060516
 imbalance_12_2: - p_DA[12] - pos_imbalance[12,2] + neg_imbalance[12,2]
   = -36.25847104981572
 imbalance_12_3: - p_DA[12] - pos_imbalance[12,3] + neg_imbalance[12,3]
   = -108.9629651646653
 imbalance_12_4: - p_DA[12] - pos_imbalance[12,4] + neg_imbalance[12,4]
   = -155.009214124361
 imbalance_13_1: - p_DA[13] - pos_imbalance[13,1] + neg_imbalance[13,1]
   = -56.73091784567828
 imbalance_13_2: - p_DA[13] - pos_imbalance[13,2] + neg_imbalance[13,2]
   = -41.27853406253715
 imbalance_13_3: - p_DA[13] - pos_imbalance[13,3] + neg_imbalance[13,3]
   = -141.4754488170254
 imbalance_13_4: - p_DA[13] - pos_imbalance[13,4] + neg_imbalance[13,4]
   = -168.7598085839971
 imbalance_14_1: - p_DA[14] - pos_imbalance[14,1] + neg_imbalance[14,1]
   = -56.1767328498395
 imbalance_14_2: - p_DA[14] - pos_imbalance[14,2] + neg_imbalance[14,2]
   = -50.374063726073
 imbalance_14_3: - p_DA[14] - pos_imbalance[14,3] + neg_imbalance[14,3]
   = -149.7060397099037
 imbalance_14_4: - p_DA[14] - pos_imbalance[14,4] + neg_imbalance[14,4]
   = -205.5729104743788
 imbalance_15_1: - p_DA[15] - pos_imbalance[15,1] + neg_imbalance[15,1]
   = -57.58218404470337
 imbalance_15_2: - p_DA[15] - pos_imbalance[15,2] + neg_imbalance[15,2]
   = -62.41320889311616
 imbalance_15_3: - p_DA[15] - pos_imbalance[15,3] + neg_imbalance[15,3]
   = -114.7238734989894
 imbalance_15_4: - p_DA[15] - pos_imbalance[15,4] + neg_imbalance[15,4]
   = -238.4509868029961
 imbalance_16_1: - p_DA[16] - pos_imbalance[16,1] + neg_imbalance[16,1]
   = -67.33147069313993
 imbalance_16_2: - p_DA[16] - pos_imbalance[16,2] + neg_imbalance[16,2]
   = -91.49610628938295
 imbalance_16_3: - p_DA[16] - pos_imbalance[16,3] + neg_imbalance[16,3]
   = -90.08367019379385
 imbalance_16_4: - p_DA[16] - pos_imbalance[16,4] + neg_imbalance[16,4]
   = -200.6488526929022
 imbalance_17_1: - p_DA[17] - pos_imbalance[17,1] + neg_imbalance[17,1]
   = -92.31021876114612
 imbalance_17_2: - p_DA[17] - pos_imbalance[17,2] + neg_imbalance[17,2]
   = -123.6294733087623
 imbalance_17_3: - p_DA[17] - pos_imbalance[17,3] + neg_imbalance[17,3]
   = -105.763880632505
 imbalance_17_4: - p_DA[17] - pos_imbalance[17,4] + neg_imbalance[17,4]
   = -230.2844489359173
 imbalance_18_1: - p_DA[18] - pos_imbalance[18,1] + neg_imbalance[18,1]
   = -111.0356973011532
 imbalance_18_2: - p_DA[18] - pos_imbalance[18,2] + neg_imbalance[18,2]
   = -155.0014861490905
 imbalance_18_3: - p_DA[18] - pos_imbalance[18,3] + neg_imbalance[18,3]
   = -140.0332897396267
 imbalance_18_4: - p_DA[18] - pos_imbalance[18,4] + neg_imbalance[18,4]
   = -253.6327428367613
 imbalance_19_1: - p_DA[19] - pos_imbalance[19,1] + neg_imbalance[19,1]
   = -117.3767982403995
 imbalance_19_2: - p_DA[19] - pos_imbalance[19,2] + neg_imbalance[19,2]
   = -171.3957912257758
 imbalance_19_3: - p_DA[19] - pos_imbalance[19,3] + neg_imbalance[19,3]
   = -189.3041849958388
 imbalance_19_4: - p_DA[19] - pos_imbalance[19,4] + neg_imbalance[19,4]
   = -252.9752704791345
 imbalance_20_1: - p_DA[20] - pos_imbalance[20,1] + neg_imbalance[20,1]
   = -104.9569016763762
 imbalance_20_2: - p_DA[20] - pos_imbalance[20,2] + neg_imbalance[20,2]
   = -182.5609321127095
 imbalance_20_3: - p_DA[20] - pos_imbalance[20,3] + neg_imbalance[20,3]
   = -266.474408512662
 imbalance_20_4: - p_DA[20] - pos_imbalance[20,4] + neg_imbalance[20,4]
   = -275.4306860064202
 imbalance_21_1: - p_DA[21] - pos_imbalance[21,1] + neg_imbalance[21,1]
   = -87.99473903221971
 imbalance_21_2: - p_DA[21] - pos_imbalance[21,2] + neg_imbalance[21,2]
   = -214.3383664249197
 imbalance_21_3: - p_DA[21] - pos_imbalance[21,3] + neg_imbalance[21,3]
   = -313.9075318035905
 imbalance_21_4: - p_DA[21] - pos_imbalance[21,4] + neg_imbalance[21,4]
   = -281.6601771489716
 imbalance_22_1: - p_DA[22] - pos_imbalance[22,1] + neg_imbalance[22,1]
   = -65.20122458685056
 imbalance_22_2: - p_DA[22] - pos_imbalance[22,2] + neg_imbalance[22,2]
   = -220.8001426703127
 imbalance_22_3: - p_DA[22] - pos_imbalance[22,3] + neg_imbalance[22,3]
   = -348.6392818927595
 imbalance_22_4: - p_DA[22] - pos_imbalance[22,4] + neg_imbalance[22,4]
   = -279.8078409226014
 imbalance_23_1: - p_DA[23] - pos_imbalance[23,1] + neg_imbalance[23,1]
   = -50.5373915111164
 imbalance_23_2: - p_DA[23] - pos_imbalance[23,2] + neg_imbalance[23,2]
   = -217.9165675900607
 imbalance_23_3: - p_DA[23] - pos_imbalance[23,3] + neg_imbalance[23,3]
   = -361.4365117108549
 imbalance_23_4: - p_DA[23] - pos_imbalance[23,4] + neg_imbalance[23,4]
   = -285.6965580787065
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
