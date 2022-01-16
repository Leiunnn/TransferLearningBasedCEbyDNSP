%% Fun_Train_Seq_Gen20120501.m 训练序列生成 参考文献 《2004_Channel estimation using implicit training》
% 20120501, modified by Henry Cinque. 
% 输入训练序列长度,产生功率为1的训练序列
% 该训练序列在时域和频域都是恒定的,是最佳的么?有这样的序列吗?
% 修改了m序列对应的星座图，只对应0和3符合，见QPSK调制的星座图可知

%************* Type 码类型说明 *********************%
% 1 频域各个频点等值，根据文献《2004_Channel estimation using implicit training》
% 2 文献《2004_Channel estimation using implicit training》给出的最优PAPR的?
% 3 chirp序列 根据《2003_A First-Order Statistical Method for Channel Estimation》
% 4 m序列 根据《2004_基于隐训练序列的信道估计与跟踪》



function [c] = Fun_Train_Seq_Gen20120501(Length, Type);


switch Type
    case 1 %Low PAPR ,Equal magnititude of each frequency piont according to paper 《2004_Channel estimation using implicit training》
        n=1:1:Length;
        if mod(Length,2)==0 %even
            c(n)=exp(j*pi.*(n-1).*(n-1)/Length); %恒定频域幅度的训练序列
        else
            c(n)=exp(j*pi.*(n-1).*n/Length); %恒定频域幅度的训练序列
        end
        return;
        
        
    case 2 %Equal magnititude of each frequency piont according to paper 《2004_Channel estimation using implicit training》
        % 此模式下文献只给出了6阶的训练序列，因此Length参数此时无用,注意 !!! 20120415
        Degree2RadianRatio = 180/pi;
        CC = sqrt(6)*[1,exp(j*35/Degree2RadianRatio),exp(j*50/Degree2RadianRatio),1,exp(-j*50/Degree2RadianRatio),exp(-j*35/Degree2RadianRatio)];
        c = ifft(CC,6);
        return;
        
        
    case 3 % chirp sequence to paper 《2003_A First-Order Statistical Method for Channel Estimation》
        % 注意chirp序列只能加到I路或Q路上,这样星座图的失真小些 Henry Cinque,20120415
        %c = [sqrt(Length) zeros(1,Length-1)];
        %chirp训练如果同时加到I,Q两路上,BER及MSE性能急剧下降  Henry Cinque,20120415
       c = [sqrt(Length)*exp(j*pi/4) zeros(1,Length-1)];
        return;
        
        
        
    case 4 % length of 6, m sequence generation and QPSK modulation
%         a=[1 0 0 1 0 1 0 0 0 0 0 1];
%         b=[0 0 0 0 0 0 0 0 0 0 0 1];
        a=[1 0 0 0 0 1];
        b=[0 0 0 0 0 1];
% % case 5 % length of 2, m sequence generation and QPSK modulation
%     a=[1 1];
%     b=[0 1];
    r=length(a);
    n=2^(r)-1;
    for i=1:n
        f(i)=b(r);
        x=a.*b;
        x1=mod(sum(x),2);    %往第n个移位寄存器输入的值
        for m=r:-1:2
            b(m)=b(m-1);
        end
        b(1)=x1;
    end
    f=transpose(f);
    Sym_Out = zeros(n,1); 
    [a1,b1]=find(f==0);
    Sym_Out(a1,:)=kron(ones(length(a1),1),0);
    [a1,b1]=find(f==1);
    Sym_Out(a1,:)=kron(ones(length(a1),1),3); 
    M_start = 0;
    PN_Alpha4 = Sym_Out((1+M_start):(6+M_start),1);
%     kk=0;
%     for M_start  = 1:1:55;%测试过的 HCQ 20120417 
%         kk = kk + 1;
%         for ii = 0:1:5
%             PN_Alpha4(ii+1) = Sym_Out(ii+M_start);
%         end  
%         c = PN_Alpha4;
%         for ShifterN = 0:1:5
%             P_AuotCorr(ShifterN+1)=pcorrelate(c,c,ShifterN);
%         end 
%         figure(kk);
%         plot(P_AuotCorr);  
%     end
    

    [data c]=Fun_QPSK_GrayMod(PN_Alpha4);
    c = transpose(c);
  return;
end
