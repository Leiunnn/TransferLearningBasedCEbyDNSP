function [S_wav,S_wav_theo] = MMSE_equali(I,J,Y,N,K,tao,P,Noise_Power,W,H_ls,H_hat)
 global alpha Pilots;
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
        
        temp1 = (I-J)*G*Y + 1./(1-alpha) .*J*(G*Y - J*Pilots);
        
        
        S_wav = W'*temp1;

        
%---理论MMSE均衡---------------
        Rv = (Noise_Power)*(I-J);
        G_theo = pinv(Rv+H_hat'*H_hat)*H_hat';
        temp2 =  (I-J)*G_theo*Y + 1./(1-alpha) .*J*(G_theo*Y - J*Pilots);
        S_wav_theo = W'*temp2;
