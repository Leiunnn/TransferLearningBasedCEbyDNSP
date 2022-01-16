function [H_source,Y] = through_channel(N, Noise_Power, X, L, F, h_onmi)


global test_time ;       
%         jjj = 1:1:L;
%         Amp_h = exp(-0.2*(jjj.'-1));
%         h_r = sqrt(0.5)*(randn(L,test_time)+1i*randn(L,test_time));
%         h_smi = h_r.*Amp_h(:,ones(1,test_time));
%         H_source = F(:,1:1:L)*h_smi;
%         for iii=1:test_time
%             H_source(:,iii) = H_source(:,iii)/norm(H_source(:,iii))*sqrt(N);
%         end
        
        H_source = squeeze(h_onmi);
        for iii=1:test_time
            H_source(:,iii) = H_source(:,iii)/norm(H_source(:,iii))*sqrt(N);
        end

        
        
        V = sqrt(0.5*Noise_Power) .*  ( randn(N,test_time) + 1i * randn(N,test_time) );
        
        %---¹ýÐÅµÀ--------
        Y = H_source  .*  X + V;