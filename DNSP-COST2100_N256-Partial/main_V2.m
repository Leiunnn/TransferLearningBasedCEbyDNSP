%======================   ��������   ====================
%       ʱ�䣺2021��1��24��
%       ���룺���ڱ�д
%       �ο����ף�G. Dou, C. He, C. Li, and J. Gao, ��Channel estimation and 
%       symbol detection for OFDM systems using data-nulling superimposed 
%       pilots,�� Electron. Lett., vol. 50, no. 3, pp. 179�C180, Jan. 2014.
%==================================================
clear;
clc;
close all;
warning off;




%--- �������� -------------
global test_time N;   
        N = 64;  %---���ز�����
        L = 20;   %---�ྶ��
        P = 6;   %---��������
        Rho = 0.2; %---���ʱ�ֵ��training-to-information power ratio
        K = round(N / P);  %---�ظ����ٴΡ��������� ��ҪKΪ������
        c_train = Train_Seq_Gen(P,1);      %��������һ�γ�ΪP�ĵ������У���(P,3)ѡ��chirp����
        test_time  = 16;
           
        
%---����Ҷ�任����------
        Fp = (1/sqrt(P))*dftmtx(P);
        F = (1/sqrt(N))*dftmtx(N);
        

global Pilot_Power;   
SNR_dB = 10:5:20; 
Sig_Power = 1e0;
Pilot_Power = Sig_Power.*Rho*ones(length(SNR_dB),1);
Data_Power = Sig_Power.*(1-Rho)*ones(length(SNR_dB),1);

Noise_Power = Sig_Power .* 10.^(-0.1*SNR_dB.');
        
%--- �����Ͼ���W,����W*W'Ϊ��λ��---       
        temp_XX=magic(N);
        [W,S,V] = svd(temp_XX);   
        
%--- diagonal matrix J ����  + ��λ��I ���� ---
        [tao,J] = matrix_gen(K,N);
        I = eye(N);   %---���ɵ�λ��I��
          
%---��������------------
        iter_time = 0;
        
%---��ȡCOST2100�������ŵ�״̬��Ϣ------------        
        load('H_Fre_domain_MPCs2.mat');
%         for ixix = 1:312
%             H_OMNI(ixix,:,:) = H_Fre(:,:,64*(ixix-1)+1:1:64*ixix);
%         end
        Num_temp = size(H_Fre); Num_ = round(Num_temp(1)./length(SNR_dB));
        


global H_source;  
    for cccc = 1:length(SNR_dB)
    tic
            for  nnnn = 1:Num_
            %--- ���� �����ź�X -----
                    [X,s] = Gen_Txdata(N,tao,K,I,J,W,Data_Power(cccc),Pilot_Power(cccc),c_train);
                    s_save(nnnn,:,:) = s;
            %--- �ź�X���ŵ�(����+AWGN) ---     
                    [H_source,Y] = through_channel(N,Noise_Power(cccc),X,L,F,H_Fre(Num_*(cccc - 1)+nnnn,:,:));
                    H_source_save(nnnn,:,:) = H_source;
                    Y_save(nnnn,:,:) = Y;
            %---LS�ŵ�����-------------------
                    [H_ls,H_mmse] = channel_esi(tao,K,c_train,Noise_Power(cccc),N,P,F,Y);
                    H_ls_save(nnnn,:,:) = H_ls;
                    H_mmse_save(nnnn,:,:) = H_mmse;
                    
                    
                    for iiii = 1:test_time
                        H_LSnorm = H_ls(:,iiii)/norm(H_ls(:,iiii));
                        H_MMSEnorm = H_mmse(:,iiii)/norm(H_mmse(:,iiii));
                        H_SourceNorm = H_source(:,iiii)/norm(H_source(:,iiii));
                        temp_MSE_LS(iiii) =(norm(H_LSnorm-H_SourceNorm))^2/(norm(H_SourceNorm)^2);
                        temp_MSE_MMSE(iiii) =(norm(H_MMSEnorm-H_SourceNorm))^2/(norm(H_SourceNorm)^2);   
                    end
                    MSE_LS_nnn(nnnn) = mean(temp_MSE_LS);
                    MSE_MMSE_nnn(nnnn) = mean(temp_MSE_MMSE);

            %---ZF����   MMSE����+���--------------------LSLSLSLSLSLSLSLS
                  [S_zf111] = ZF_equali(I,J,Y,W,H_ls);
                  [After_iter_S_zf111] = Signal_Detect(S_zf111,iter_time,J,W);  %---���
                  NN_LS_Equali_ZF_save(nnnn,:,:) = After_iter_S_zf111;
                  
                  [LS_CE_ZF_EQ_nnn(nnnn)] = Calc_Ber(After_iter_S_zf111,s); 
            for iiii = 1:test_time
                  [S_wav111(:,iiii),S_wav_theo111(:,iiii)] = MMSE_equali(I,J,Y(:,iiii),N,K,tao,P,Noise_Power(cccc),W,H_ls(:,iiii),diag(H_ls(:,iiii))); %---MMSE���� 
            end  
                  [After_iter_data111] = Signal_Detect(S_wav111,iter_time,J,W);  %---���
                  [After_iter_data_theo111] = Signal_Detect(S_wav_theo111,iter_time,J,W);  %---���
                  [LS_CE_MMSE_EQ_nnn(nnnn)] = Calc_Ber(After_iter_data111,s);       
                  [LS_CE_MMSE_EQ_theo_nnn(nnnn)] = Calc_Ber(After_iter_data_theo111,s);  


            %---ZF����   MMSE����+���--------------------MMSEMMSEMMSEMMSE
                  [S_zf222] = ZF_equali(I,J,Y,W,H_mmse);
                  [After_iter_S_zf222] = Signal_Detect(S_zf222,iter_time,J,W);  %---���
                  NN_MMSE_Equali_ZF_save(nnnn,:,:) = After_iter_S_zf222;
                  
                  [MMSE_CE_ZF_EQ_nnn(nnnn)] = Calc_Ber(After_iter_S_zf222,s); 
            for iiii = 1:test_time
                  [S_wav222(:,iiii),S_wav_theo222(:,iiii)] = MMSE_equali(I,J,Y(:,iiii),N,K,tao,P,Noise_Power(cccc),W,H_mmse(:,iiii),diag(H_mmse(:,iiii))); %---MMSE���� 
            end  
                  [After_iter_data222] = Signal_Detect(S_wav222,iter_time,J,W);  %---���
                  [After_iter_data_theo222] = Signal_Detect(S_wav_theo222,iter_time,J,W);  %---���
                  [MMSE_CE_MMSE_EQ_nnn(nnnn)] = Calc_Ber(After_iter_data222,s);       
                  [MMSE_CE_MMSE_EQ_theo_nnn(nnnn)] = Calc_Ber(After_iter_data_theo222,s);  


            %---ZF����   MMSE����+���--------------------HHHHHHHHHHHHHHHHHHHHHHHHH
                  [S_zf333] = ZF_equali(I,J,Y,W,H_source);
                  [After_iter_S_zf333] = Signal_Detect(S_zf333,iter_time,J,W);  %---���
                  NN_H_Equali_ZF_save(nnnn,:,:) = After_iter_S_zf333;
                  
                  [H_CE_ZF_EQ_nnn(nnnn)] = Calc_Ber(After_iter_S_zf333,s); 
            for iiii = 1:test_time
                  [S_wav333(:,iiii),S_wav_theo333(:,iiii)] = MMSE_equali(I,J,Y(:,iiii),N,K,tao,P,Noise_Power(cccc),W,H_source(:,iiii),diag(H_source(:,iiii))); %---MMSE���� 
            end  
                  [After_iter_data333] = Signal_Detect(S_wav333,iter_time,J,W);  %---���
                  [After_iter_data_theo333] = Signal_Detect(S_wav_theo333,iter_time,J,W);  %---���
                  [H_CE_MMSE_EQ_nnn(nnnn)] = Calc_Ber(After_iter_data333,s);       
                  [H_CE_MMSE_EQ_theo_nnn(nnnn)] = Calc_Ber(After_iter_data_theo333,s);  

            end
    toc
            MSE_LS(cccc) = mean(MSE_LS_nnn);
            MSE_MMSE(cccc) = mean(MSE_MMSE_nnn);
            LS_CE_ZF_EQ(cccc) = mean(LS_CE_ZF_EQ_nnn);
            LS_CE_MMSE_EQ(cccc) = mean(LS_CE_MMSE_EQ_nnn);
            LS_CE_MMSE_EQ_theo(cccc) = mean(LS_CE_MMSE_EQ_theo_nnn);
            MMSE_CE_ZF_EQ(cccc) = mean(MMSE_CE_ZF_EQ_nnn);
            MMSE_CE_MMSE_EQ(cccc) = mean(MMSE_CE_MMSE_EQ_nnn);
            MMSE_CE_MMSE_EQ_theo(cccc) = mean(MMSE_CE_MMSE_EQ_theo_nnn);
            H_CE_ZF_EQ(cccc) = mean(H_CE_ZF_EQ_nnn);
            H_CE_MMSE_EQ(cccc) = mean(H_CE_MMSE_EQ_nnn);
            H_CE_MMSE_EQ_theo(cccc) = mean(H_CE_MMSE_EQ_theo_nnn);
            
            
%             %---������������ѵ������---
%             save('teData/L_8/P_8/teNN_H_LS'+string(SNR_dB(cccc))+'.mat','H_ls_save');                        %������LS���ƺ�õ���CSI
%             save('teData/L_8/P_8/teNN_H_MMSE'+string(SNR_dB(cccc))+'.mat','H_mmse_save'); 
%             save('teData/L_8/P_8/teNN_TabH'+string(SNR_dB(cccc))+'.mat','H_source_save');                    %����ԴCSI
%             save('teData/L_8/P_8/teNN_Rxdata'+string(SNR_dB(cccc))+'.mat','Y_save');                         %������������
%             save('teData/L_8/P_8/teNN_LS_Equali_ZF'+string(SNR_dB(cccc))+'.mat','NN_LS_Equali_ZF_save');     %����LS���ƺ�ZF����
%             save('teData/L_8/P_8/teNN_MMSE_Equali_ZF'+string(SNR_dB(cccc))+'.mat','NN_MMSE_Equali_ZF_save'); %����MMSE���ƺ�ZF����
%             save('teData/L_8/P_8/teNN_H_Equali_ZF'+string(SNR_dB(cccc))+'.mat','NN_H_Equali_ZF_save');       %����Դ�ŵ����ƺ�ZF����
%             save('teData/L_8/P_8/teNN_Tabdata'+string(SNR_dB(cccc))+'.mat','s_save');                        %�������͵ľ�QPSK��������

    end

    figure;
    semilogy(SNR_dB,MSE_LS,'--go',SNR_dB,MSE_MMSE,'--ks','LineWidth',2,'MarkerSize',8);
    xlabel('SNR(dB)');
    ylabel('NMSE');
    legend('LS method','MMSE method');
    grid on;
    
    
    figure;
    semilogy(SNR_dB,LS_CE_ZF_EQ,':xC',...
            SNR_dB,LS_CE_MMSE_EQ,'-.ob',...
            SNR_dB,MMSE_CE_ZF_EQ,'-+g',...
            SNR_dB,H_CE_ZF_EQ,':sm',...
            SNR_dB,MMSE_CE_MMSE_EQ,':+r',...
            SNR_dB,H_CE_MMSE_EQ,'-xk',...
        'LineWidth',2,'MarkerSize',8);
    xlabel('SNR(dB)');
    ylabel('BER');
    legend('LS\_CE + ZF\_EQ',...
        'LS\_CE + MMSE\_EQ',...
        'MMSE\_CE + ZF\_EQ',...
        'H\_s + ZF\_EQ',...
        'MMSE\_CE + MMSE\_EQ',...
        'H\_s + MMSE\_EQ');
    grid on;

    

    
    
    %     %---����NMSE----------------------
%     for iiii = 1:test_time
%             H_LSnorm = H_ls(:,iiii)/norm(H_ls(:,iiii));
%             H_MMSEnorm = H_mmse(:,iiii)/norm(H_mmse(:,iiii));
%             H_SourceNorm = H_source(:,iiii)/norm(H_source(:,iiii));
%             temp_MSE_LS(iiii) =(norm(H_LSnorm-H_SourceNorm))^2/(norm(H_SourceNorm)^2);
%             temp_MSE_MMSE(iiii) =(norm(H_MMSEnorm-H_SourceNorm))^2/(norm(H_SourceNorm)^2);   
%     end
    


%     %----NMSE���--------------
%             MSE_LS(cccc) = mean(temp_MSE_LS);
%             MSE_MMSE(cccc) = mean(temp_MSE_MMSE);
%             MSE_LS_theo(cccc) = L * Noise_Power(cccc)/(Pilot_Power(cccc)*(norm(c_train))^2);

%     figure;
%     semilogy(SNR_dB,MSE_LS,'--go',SNR_dB,MSE_LS_theo,'--r+',SNR_dB,MSE_MMSE,'--ks','LineWidth',2,'MarkerSize',8);
%     xlabel('SNR(dB)');
%     ylabel('NMSE');
%     legend('LS method','LS method theogy');
%     grid on;












