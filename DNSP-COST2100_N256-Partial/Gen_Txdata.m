



function [X,s] = Gen_Txdata(N,tao,K,I,J,W,Data_Power,Pilot_Power,c_train)
global test_time Pilots alpha c_symbol;    



        s = sqrt(0.5) * ( (2*randi([0, 1], N,test_time )-1) + 1i * (2*randi([0, 1], N,test_time )-1) );  %---生成数据
        c = zeros(N,1);
        c(tao:K:end, 1) = c_train;
        Pilots = c;
        c_symbol =  c(:,ones(1,test_time));

        
        X = sqrt(Data_Power)*(I- alpha.*J) * W *  s + sqrt(Pilot_Power)*J * c_symbol;
        
        

        
        
