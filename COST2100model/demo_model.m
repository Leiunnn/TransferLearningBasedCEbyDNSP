%% Demo model to run the COST 2100 channel model
% This is an example file of how to run the model. The input parameters, as
% stated below, are first chosen before the main function cost2100 run. The
% result from cost2100 is combined with different antenna patterns, with
% the available options stated as Output below.
%
% For more information about the available networks, references and version
% history, see the file 'Readme.txt'. If you use the COST 2100 channel model
% for publications, please refer to the stated publications.
%
%------
%Input:
%------
% Network : 'IndoorHall_5GHz','SemiUrban_300MHz','Indoor_CloselySpacedUser_2_6GHz','SemiUrban_CloselySpacedUser_2_6GHz', or 'SemiUrban_VLA_2_6GHz'
% Band : 'Wideband' or 'Narrowband'
% Link: 'Multiple' or 'Single'
% Antenna: 'SISO_omni', 'MIMO_omni', 'MIMO_dipole', 'MIMO_measured', 'MIMO_Cyl_patch', 'MIMO_VLA_omni'
% scenario: 'LOS' or 'NLOS'        
% freq: Frequency band [Hz]
% snapRate: Number of snapshots per s
% snapNum: Number of simulated snapshots         
% BSPosCenter: Center position of BS array [x, y, z] [m]
% BSPosSpacing: Inter-position spacing [m], for large arrays
% BSPosNum: Number of positions at each BS site, for large arrays
% MSPos: Position of MSs [m]
% MSVelo: Velocity of MSs [m/s]
%
%------
%Output:
%------ 
% 1) SISO_omni: Transfer function for SISO omni-directional antenna
% create_IR_omni: users have to set up the frequency separation, delta_f
% 
% 2) MIMO_omni: Transfer function for MIMO omini-directional antenna
% create_IR_omni_MIMO: users have to set up the frequency separation, delta_f.
% Only 2 by 2 MIMO system is implemented.
% 
% 3) MIMO_dipole: Transfer function for a theoretical antenna response for 
% any size of lambda/2-spaced linear dipole antenna arrays. An Ntx-by-Nrx theoretical 
% antenna array response is generated and the correponding 
% channel transfer function is simulated.
% 
% 4) MIMO_measured: Transfer function for any measured MIMO antenna response
% get_H: users have to provide the full antenna response at the BS and 
% MS sides, and also the rotation of the antenna arrays. The antenna 
% response mat file have to be the same format as 'antSample.mat' file.
%
% 5) MIMO_Cyl_patch: Transfer function for a synthetic pattern of a cylindrical array 
% with 128 antennas.
% get_IR_Cyl_patch: users have to set up the frequency separation, delta_f, as well as
% provide the full antennas response at the BS and MS sides. The antennas response mat 
% file have to be the same format as the 'BS_Cyl_AntPattern.mat' and 'MS_AntPattern_User.mat' 
% files.

% 6) MIMO_VLA_omni: Transfer function for a physically large array with 128 omni-directional 
% antennas, with lambda/2 inter-element separation, and MS with omni-directional antenna.
% create_IR_omni_MIMO_VLA: users have to set up the frequency separation, delta_f

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This file is a part of the COST2100 channel model.
%
%This program, the COST2100 channel model, is free software: you can 
%redistribute it and/or modify it under the terms of the GNU General Public 
%License as published by the Free Software Foundation, either version 3 of 
%the License, or (at your option) any later version.
%
%This program is distributed in the hope that it will be useful, but 
%WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
%or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
%for more details.
%
%If you use it for scientific purposes, please consider citing it in line 
%with the description in the Readme-file, where you also can find the 
%contributors.
%
%You should have received a copy of the GNU General Public License along 
%with this program. If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;
close all;
warning off;

% Choose a Network type out of 
% {'IndoorHall_5GHz','SemiUrban_300MHz','Indoor_CloselySpacedUser_2_6GHz','SemiUrban_CloselySpacedUser_2_6GHz','SemiUrban_VLA_2_6GHz'}
% to parameterize the COST2100 model
Network = 'SemiUrban_300MHz';
% In COST2100, # links = # BSs x # MSs
% Set Link type to `Multiple' if you work with more than one link
% Set Link type to `Single' otherwise
Link = 'Single';
% Choose an Antenna type out of
% {'SISO_omni', 'MIMO_omni', 'MIMO_dipole', 'MIMO_measured', 'MIMO_Cyl_patch', 'MIMO_VLA_omni'}
Antenna = 'SISO_omni';
% ...and type of channel: {'Wideband','Narrowband'}.
Band = 'Wideband';

% Here are some tested combinations of the above variables:
% 'IndoorHall_5GHz', 'Single', 'SISO_omni', 'Wideband'
% 'SemiUrban_300MHz', 'Single', 'SISO_omni', 'Wideband'
% 'SemiUrban_300MHz', 'Multiple', 'MIMO_omni', 'Wideband'
% 'Indoor_CloselySpacedUser_2_6GHz', 'Multiple', 'MIMO_Cyl_patch', 'Wideband'
% 'SemiUrban_CloselySpacedUser_2_6GHz', 'Multiple', 'MIMO_Cyl_patch', 'Wideband'
% 'SemiUrban_VLA_2_6GHz', 'Single', 'MIMO_VLA_omni', 'Wideband'
% 'SemiUrban_VLA_2_6GHz', 'Multiple', 'MIMO_VLA_omni', 'Wideband'
global F N L;
N = 256;
F=(1/sqrt(N))*dftmtx(N);
snap_Num = 16;
L = 6;
NumOfSets = 10000;
for cccc = 1:1:NumOfSets
%     L = unidrnd(25);
    switch Network
        %%%%%%%%%%%%%%%%%%%%%%%
        case 'SemiUrban_300MHz'
        %%%%%%%%%%%%%%%%%%%%%%%
            switch Link
               case 'Single'
                    scenario = 'NLOS'; % {'LOS', 'NLOS'} is available
                    freq = [286e6 286.6e6]; %  [Hz]
                    snapRate = 6e6; % Number of snapshots per s  400 0.35   312 0.45
                    snapNum = snap_Num; % Number of snapshots        
                    MSPos  = [100 0 -200]; % [m]
                    MSVelo = [1 0 0]; % [m/s]
                    BSPosCenter  = [0 0 0]; % Center position of BS array [x, y, z] [m]
                    BSPosSpacing = [0 0 0]; % Inter-position spacing (m), for large arrays
                    BSPosNum = 1; % Number of positions at each BS site, for large arrays
            end
    end
    tic
    %% Get the MPCs from the COST 2100 channel model
    [...
        paraEx,...       % External parameters
        paraSt,...       % Stochastic parameters
        link,...         % Simulated propagation data for all links [nBs,nMs]
        env...           % Simulated environment (clusters, clusters' VRs, etc.)
    ] = cost2100...
    (...
        Network,...      % Model environment
        scenario,...     % LOS or NLOS
        freq,...         % [starting freq., ending freq.]
        snapRate,...     % Number of snapshots per second
        snapNum,...      % Total # of snapshots
        BSPosCenter,...  % Center position of each BS
        BSPosSpacing,... % Position spacing for each BS (parameter for physically very-large arrays)
        BSPosNum,...     % Number of positions on each BS (parameter for physically very-large arrays)
        MSPos,...        % Position of each MS
        MSVelo...        % Velocity of MS movements
        );         
    toc

    %% Visualize the generated environment
    % if 1  
    %     switch Network
    %         case {'IndoorHall_5GHz','SemiUrban_300MHz'}   
    %              visual_channel(paraEx, paraSt, link, env);
    %         case {'SemiUrban_VLA_2_6GHz','SemiUrban_CloselySpacedUser_2_6GHz','Indoor_CloselySpacedUser_2_6GHz'}   
    %              visualize_channel_env(paraEx, paraSt, link, env); axis equal; view(2);
    %     end   
    % end

    %% Ccombine propagation data with antenna patterns
    % Construct the channel data
    % The following is example code
    % End users can write their own code

    switch Antenna
        %%%%%%%%%%%%%%%%%%
        case 'SISO_omni' % SISO between one BS and one MS
        %%%%%%%%%%%%%%%%%%
            switch Link
                %%%%%%%%%%%%%%
                case 'Single'
                %%%%%%%%%%%%%%
                    delta_f = (freq(2)-freq(1))/(N-1);
                    h_omni = create_IR_omni(link,freq,delta_f,Band);
                    H_OMNI = F*(h_omni.');
                    
            end
    end
               
    H_Fre(cccc,:,:) = H_OMNI; 
end
save('H_Fre256MPCs6_0.6Mhz.mat','H_Fre');


