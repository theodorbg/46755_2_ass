# 46755_2_ass
Assignment 2 46755 repo



# in sample scenarios
first 200 scenarios
under condition 0
Wind day 0 to 9
(price day 0 to 19) x 20 

So in other words, the first 200 scenarios are keeping the condition constant at 0, and then varies the price day from 0 to 19 for 10 wind days.

# Out of sample scenarios
last 1400 scenarios
Start at condition 0 and continues going through each price day starting from wind day 10, going through all the 20 wind days, to complete the first 400 days of condition 0. Then it moves on to condition 1, and repeats the loop for 400 new scenarios, and so on until each condition as been tried with each combination of wind days and price days.
