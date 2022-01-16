function [S_zf] = ZF_equali(I,J,Y,W,H_ls)
 
%---È¥µôµ¼Òý----
        Y_data = (I-J)*Y;
        G_zf = 1./H_ls;
        S_zf = W'*(G_zf.*Y_data);
        
        
        
        

