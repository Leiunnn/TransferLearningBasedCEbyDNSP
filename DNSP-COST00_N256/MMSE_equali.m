function [S_wav,S_wav_theo] = MMSE_equali(I,J,Y,N,K,tao,P,Noise_Power,W,H_ls,H_hat)
 
%---去掉导引----
        Y_data = (I-J)*Y;
        
%---MMSE均衡---------------
        g = zeros(N,1);
        for n = 1:N
            if n == tao+K*P
                g(n,1) = 1./H_ls(n,1);
            else
                g(n,1) = (conj(H_ls(n,1)))/((abs(H_ls(n,1)).^2)+(Noise_Power));
            end
        end
        
        G = diag(g);
        S_wav = W'*G*Y_data;

        
%---理论MMSE均衡---------------
        Rv = (Noise_Power)*(I-J);
        G_theo = pinv(Rv+H_hat'*H_hat)*H_hat';
        S_wav_theo = W'*G_theo*Y_data;
