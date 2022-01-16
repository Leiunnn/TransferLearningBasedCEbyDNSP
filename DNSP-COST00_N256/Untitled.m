clear all;
clc;

SNR_dB = 24:8:32; 
   for cccc = 1:length(SNR_dB)
            tic
            load('teData/L_88/P_8/teNN_H_LS'+string(SNR_dB(cccc))+'.mat','H_ls_save');                        %——经LS估计后得到的CSI
            load('teData/L_88/P_8/teNN_H_MMSE'+string(SNR_dB(cccc))+'.mat','H_mmse_save'); 
            load('teData/L_88/P_8/teNN_TabH'+string(SNR_dB(cccc))+'.mat','H_source_save');                    %——源CSI
            load('teData/L_88/P_8/teNN_Rxdata'+string(SNR_dB(cccc))+'.mat','Y_save');                         %——接收数据
            load('teData/L_88/P_8/teNN_LS_Equali_ZF'+string(SNR_dB(cccc))+'.mat','NN_LS_Equali_ZF_save');  
            load('teData/L_88/P_8/teNN_LS_Equali_MMSE'+string(SNR_dB(cccc))+'.mat','NN_LS_Equali_MMSE_save'); %——LS估计后经ZF均衡
            load('teData/L_88/P_8/teNN_MMSE_Equali_ZF'+string(SNR_dB(cccc))+'.mat','NN_MMSE_Equali_ZF_save'); %——MMSE估计后经ZF均衡
            load('teData/L_88/P_8/teNN_MMSE_Equali_MMSE'+string(SNR_dB(cccc))+'.mat','NN_MMSE_Equali_MMSE_save'); %——MMSE估计后经ZF均衡
            load('teData/L_88/P_8/teNN_H_Equali_ZF'+string(SNR_dB(cccc))+'.mat','NN_H_Equali_ZF_save');       %——源信道估计后经ZF均衡
            load('teData/L_88/P_8/teNN_Tabdata'+string(SNR_dB(cccc))+'.mat','s_save');  


            %---保存用于网络训练数据---
            save('teData/L_88/P_8/teNN_H_LS'+string(SNR_dB(cccc))+'.mat','H_ls_save');                        %——经LS估计后得到的CSI
            save('teData/L_88/P_8/teNN_H_MMSE'+string(SNR_dB(cccc))+'.mat','H_mmse_save'); 
            save('teData/L_88/P_8/teNN_TabH'+string(SNR_dB(cccc))+'.mat','H_source_save');                    %——源CSI
            save('teData/L_88/P_8/teNN_Rxdata'+string(SNR_dB(cccc))+'.mat','Y_save');                         %——接收数据
            save('teData/L_88/P_8/teNN_LS_Equali_ZF'+string(SNR_dB(cccc))+'.mat','NN_LS_Equali_ZF_save');  
            save('teData/L_88/P_8/teNN_LS_Equali_MMSE'+string(SNR_dB(cccc))+'.mat','NN_LS_Equali_MMSE_save'); %——LS估计后经ZF均衡
            save('teData/L_88/P_8/teNN_MMSE_Equali_ZF'+string(SNR_dB(cccc))+'.mat','NN_MMSE_Equali_ZF_save'); %——MMSE估计后经ZF均衡
            save('teData/L_88/P_8/teNN_MMSE_Equali_MMSE'+string(SNR_dB(cccc))+'.mat','NN_MMSE_Equali_MMSE_save'); %——MMSE估计后经ZF均衡
            save('teData/L_88/P_8/teNN_H_Equali_ZF'+string(SNR_dB(cccc))+'.mat','NN_H_Equali_ZF_save');       %——源信道估计后经ZF均衡
            save('teData/L_88/P_8/teNN_Tabdata'+string(SNR_dB(cccc))+'.mat','s_save');  
            toc
   end
