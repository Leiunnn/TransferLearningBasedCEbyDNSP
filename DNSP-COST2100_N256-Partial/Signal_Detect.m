function [After_iter_data] = Signal_Detect(S_wav,iter_time,J,W)
 switch iter_time
    
    case 0 
        S_0 = minimum_range(S_wav);
        output_data = S_0;

    case 1
        S_0 = minimum_range(S_wav);
        S_1 = minimum_range(S_wav + W'*J*W*S_0);
        output_data = S_1;
        
    case 2
        S_0 = minimum_range(S_wav);
        S_1 = minimum_range(S_wav + W'*J*W*S_0);
        S_2 = minimum_range(S_wav + W'*J*W*S_1);
        output_data = S_2;
        

 end
 After_iter_data = output_data;
