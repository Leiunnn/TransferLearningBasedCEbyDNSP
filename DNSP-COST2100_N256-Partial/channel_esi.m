

function [H_ls,H_mmse] = channel_esi(tao,K,c_train,Noise_Power,N,P,F,Y)
global test_time H_source Pilots;

%---LS信道估计-------------
        c_train = c_train.';
        c_pilots =  c_train(:,ones(1,test_time));
        Pilot_Freq = Y(tao:K:end,:);
        H_wav = Pilot_Freq./c_pilots;
        
        Fp_temp = F((1:length(Pilot_Freq(:,1))),(tao:K:end));
        h_wav = conj(Fp_temp) * H_wav;
        H_ls = F(:,1:1:P)*h_wav;
        for iii=1:1:test_time
            H_ls(:,iii) = H_ls(:,iii)/norm(H_ls(:,iii))*sqrt(N);
        end

%---MMSE信道估计-------------
        for iii=1:1:test_time
            Rgg = H_source(:,iii)*H_source(:,iii)';
            H_mmse(:,iii) = Rgg*pinv(Rgg + Noise_Power*pinv( (Pilots)'* (Pilots) ))*H_ls(:,iii);
        end

















        
        
        
%         H_hat = diag(H_ls);