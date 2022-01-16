

%************************************************************************%
%**   Zadoff_Chu.m - To construct Zadoff-Chu sequence                 ***%
%**   Copyright (c) XiHua University. All rights reserved.            ***%
%**   Purpose:                                                        ***%
%**   Create by Chaojin Qing                                          ***%
%**   2017-6-15                                                       ***%
%**   Edition 0.1                                                     ***%
%**   Security level [Private]                                        ***%
%************************************************************************%
function out = Zadoff_Chu(len_seq)
% clear;
% clc
% close all;
% len_seq = 512;
N = len_seq;
M = 1;
q = 0;
k=1:1:N;   %k从1到32，间隔为1


temp = mod(len_seq,2); %temp为len_seq的取余
if  temp==0
    out = exp(i*2*pi*M/N*(k.^2./2+q.*k));
else
    out = exp(i*2*pi*M/N*(k.*(k+1)./2+q.*k));
end;











