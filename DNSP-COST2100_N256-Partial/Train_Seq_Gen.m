%% Fun_Train_Seq_Gen20120501.m ѵ���������� �ο����� ��2004_Channel estimation using implicit training��
% 20120501, modified by Henry Cinque. 
% ����ѵ�����г���,��������Ϊ1��ѵ������
% ��ѵ��������ʱ���Ƶ���Ǻ㶨��,����ѵ�ô?��������������?
% �޸���m���ж�Ӧ������ͼ��ֻ��Ӧ0��3���ϣ���QPSK���Ƶ�����ͼ��֪

%************* Type ������˵�� *********************%
% 1 Ƶ�����Ƶ���ֵ���������ס�2004_Channel estimation using implicit training��
% 2 ���ס�2004_Channel estimation using implicit training������������PAPR��?
% 3 chirp���� ���ݡ�2003_A First-Order Statistical Method for Channel Estimation��
% 4 m���� ���ݡ�2004_������ѵ�����е��ŵ���������١�



function [c] = Fun_Train_Seq_Gen20120501(Length, Type);


switch Type
    case 1 %Low PAPR ,Equal magnititude of each frequency piont according to paper ��2004_Channel estimation using implicit training��
        n=1:1:Length;
        if mod(Length,2)==0 %even
            c(n)=exp(j*pi.*(n-1).*(n-1)/Length); %�㶨Ƶ����ȵ�ѵ������
        else
            c(n)=exp(j*pi.*(n-1).*n/Length); %�㶨Ƶ����ȵ�ѵ������
        end
        return;
        
        
    case 2 %Equal magnititude of each frequency piont according to paper ��2004_Channel estimation using implicit training��
        % ��ģʽ������ֻ������6�׵�ѵ�����У����Length������ʱ����,ע�� !!! 20120415
        Degree2RadianRatio = 180/pi;
        CC = sqrt(6)*[1,exp(j*35/Degree2RadianRatio),exp(j*50/Degree2RadianRatio),1,exp(-j*50/Degree2RadianRatio),exp(-j*35/Degree2RadianRatio)];
        c = ifft(CC,6);
        return;
        
        
    case 3 % chirp sequence to paper ��2003_A First-Order Statistical Method for Channel Estimation��
        % ע��chirp����ֻ�ܼӵ�I·��Q·��,��������ͼ��ʧ��СЩ Henry Cinque,20120415
        %c = [sqrt(Length) zeros(1,Length-1)];
        %chirpѵ�����ͬʱ�ӵ�I,Q��·��,BER��MSE���ܼ����½�  Henry Cinque,20120415
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
        x1=mod(sum(x),2);    %����n����λ�Ĵ��������ֵ
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
%     for M_start  = 1:1:55;%���Թ��� HCQ 20120417 
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
