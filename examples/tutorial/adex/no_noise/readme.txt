Steps to get zero noise:

Set input rate in adex.xml to 0
In .model file add reversal from 0,0 to 194,40:
0,0	194,40	1.0
In .mat file, replace 0,0 transition:
1000;0,0;0,0:1.0;
