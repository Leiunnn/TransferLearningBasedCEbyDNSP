function err_num = ErrNum_cmp(X_source,X_ELM)


%---输入数据必须为sqrt(0.5)的数值
%     err_num = length(find((X_source - X_ELM) ~= 0));
    real_x = real(X_source); imag_x = imag(X_source);
    real_est = real(X_ELM); imag_est = imag(X_ELM);
%    
% 
    temp_real = find(real_est ~= real_x); temp_imag = find(imag_est ~= imag_x); 
    err_num = length(temp_real)+length(temp_imag);



