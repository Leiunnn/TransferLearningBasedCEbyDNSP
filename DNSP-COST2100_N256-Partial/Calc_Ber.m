function [ber] = Calc_Ber(After_iter_data,s)
global test_time N;
err_num = ErrNum_cmp(s,After_iter_data);
ber = err_num./(2*N*test_time);

