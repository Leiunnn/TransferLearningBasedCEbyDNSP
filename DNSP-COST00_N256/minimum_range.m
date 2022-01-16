function [output_data] = minimum_range(input_data)

ttt_real = real(input_data);
ttt_real(find(ttt_real>=0)) = sqrt(0.5);
ttt_real(find(ttt_real<0)) = -sqrt(0.5);

ttt_imag = imag(input_data);
ttt_imag(find(ttt_imag>=0)) = sqrt(0.5);
ttt_imag(find(ttt_imag<0)) = -sqrt(0.5);

output_data = ttt_real + 1i * ttt_imag;