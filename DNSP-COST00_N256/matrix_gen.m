



function [tao,J] = matrix_gen(K,N)
        tao = 1;
%        tao = randi([1,K-1],1,1);  %---随机产生[0,K-1]上的取值作为频域放导引点的偏移值；
        temp_J = zeros(N,1);
        temp_J(tao:K:end, 1) = 1;
        J = diag(temp_J);