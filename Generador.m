clear;clc;
fileID = fopen('Dataset.txt','w');
x = -2:4/100:2;
y = sin((pi/4) * x);
fprintf(fileID,'%1.4f\n',x);
fclose(fileID);
fileID = fopen('Target.txt','w');
fprintf(fileID,'%1.4f\n',y);
fclose(fileID);