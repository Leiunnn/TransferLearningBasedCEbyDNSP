



function [tao,J] = matrix_gen(K,N)
        tao = 1;
%        tao = randi([1,K-1],1,1);  %---�������[0,K-1]�ϵ�ȡֵ��ΪƵ��ŵ������ƫ��ֵ��
        temp_J = zeros(N,1);
        temp_J(tao:K:end, 1) = 1;
        J = diag(temp_J);