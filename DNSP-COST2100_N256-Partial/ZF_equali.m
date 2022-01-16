function [S_zf] = ZF_equali(I,J,Y,W,H_ls)
 global alpha Pilots c_symbol;
%---È¥µôµ¼Òý----
        Y_data = (I-J)*Y;
        
       
        
        G_zf = 1./H_ls;
        
        
         temp = (I-J)*G_zf.*Y + 1./(1-alpha) .*J*(G_zf.*Y - J*c_symbol);
         
         
        S_zf = W'*temp;
        
        
        
        

