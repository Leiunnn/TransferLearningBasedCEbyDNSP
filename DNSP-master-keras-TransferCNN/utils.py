from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import numpy as np
import h5py


def split_RealImage_3Dim(dim1,dim2,dim3,x):

    x_dim = np.zeros((dim1,dim2,dim3,2))
    x_dim[:,:,:,0] = x['real']
    x_dim[:,:,:,1] = x['imag']
    x_dim = np.concatenate((x_dim[:, :, :,0], x_dim[:, :,:, 1]),axis=0).reshape(2*dim1, dim2, dim3, 1)

    return x_dim


def split_RealImage_2Dim(dim1,dim2,x,flag):

    if flag == 'v7':
        x_dim = np.zeros((dim1,dim2,2))
        x_dim[:,:,0] = x['real']
        x_dim[:,:,1] = x['imag']
        x_dim = np.concatenate((x_dim[:, :, 0], x_dim[:, :, 1]),axis=0)

    else:
        x_dim = np.zeros((dim1,dim2,2))
        x_dim[:, :, 0] = np.real(x)
        x_dim[:, :, 1] = np.imag(x)
        x_dim = np.concatenate((x_dim[:, :, 0], x_dim[:, :, 1]), axis=0)

    return x_dim

# 加载训练数据
def load_Trdatasets(carrier_num,SNR):
    print('Load Datasets ...')

    # --经过LS估计到的信道
    Pre_LSchannel = h5py.File('trData/L_8/P_8/' + 'trNN_H_LS8.mat')['H_ls_save'][:].T
    for snr in SNR:
        Temp_LS = h5py.File('trData/L_8/P_8/' + 'trNN_H_LS'+str(snr+8)+'.mat')['H_ls_save'][:].T
        Pre_LSchannel = np.concatenate((Pre_LSchannel,Temp_LS),axis=0)

    # --信道标签
    H_source = h5py.File('trData/L_8/P_8/' + 'trNN_TabH8.mat')['H_source_save'][:].T
    for snr in SNR:
        Temp_H = h5py.File('trData/L_8/P_8/' + 'trNN_TabH'+str(snr+8)+'.mat')['H_source_save'][:].T
        H_source = np.concatenate((H_source,Temp_H),axis=0)

    # --接收信号
    Rx_data = h5py.File('trData/L_8/P_8/' + 'trNN_Rxdata8.mat')['Y_save'][:].T    # --接收到的信号
    for snr in SNR:
        Temp_Rx = h5py.File('trData/L_8/P_8/' + 'trNN_Rxdata'+str(snr+8)+'.mat')['Y_save'][:].T
        Rx_data = np.concatenate((Rx_data,Temp_Rx),axis=0)

    # --接收信号
    Tab_data = h5py.File('trData/L_8/P_8/' + 'trNN_Tabdata8.mat')['s_save'][:].T # --发送的QPSK信号
    for snr in SNR:
        Temp_tab = h5py.File('trData/L_8/P_8/' + 'trNN_Tabdata'+str(snr+8)+'.mat')['s_save'][:].T
        Tab_data = np.concatenate((Tab_data,Temp_tab),axis=0)


    #---经过LS预估计的信道分成实部虚部并拼接
    h_ls = np.zeros((len(Pre_LSchannel),carrier_num,len(Pre_LSchannel.T),2))
    h_ls[:,:,:,0] = Pre_LSchannel['real']
    h_ls[:,:,:,1] = Pre_LSchannel['imag']
    h_ls = np.concatenate((h_ls[:, :, :,0], h_ls[:, :,:, 1]),axis=0).reshape(2*len(Pre_LSchannel), carrier_num, len(Pre_LSchannel.T), 1)

    # ---原始信道的信道分成实部虚部并拼接
    h_tab = np.zeros((len(H_source),carrier_num,len(H_source.T),2))
    h_tab[:,:,:,0] = H_source['real']
    h_tab[:,:,:,1] = H_source['imag']
    h_tab = np.concatenate((h_tab[:,:,:,0] , h_tab[:,:,:,1] ), axis=0).reshape(2*len(H_source), carrier_num, len(H_source.T), 1)
    h_tab = h_tab.reshape((2*len(H_source), len(H_source.T)* 256,1), order='F')
    h_tab = h_tab.transpose(0,2,1)
    # print(perfect[0:5, 0:5])
    # ---发送的经过调制后的信号分成实部虚部并拼接
    s_tab = np.zeros((len(Tab_data),carrier_num,len(Tab_data.T),2))
    s_tab[:,:,:,0] = Tab_data['real']
    s_tab[:,:,:,1] = Tab_data['imag']
    s_tab = np.concatenate((s_tab[:,:,:,0], s_tab[:,:,:,1]), axis=0).reshape(2*len(Pre_LSchannel), carrier_num, len(Pre_LSchannel.T), 1)

    # ---训练集，校验集以及测试集的划分
    data_tr, data_check, label_tr, label_check = train_test_split(h_ls, h_tab, test_size=0.4)
    data_val, data_te, label_val, label_te = train_test_split(data_check, label_check, test_size=0.5)

    return data_tr, label_tr, data_val, label_val, data_te, label_te

# 加载微调数据数据
def load_Finedatasets(carrier_num,multi_path,SNR):
    print('Load Datasets ...')

    # --经过LS估计到的信道
    Pre_LSchannel = h5py.File('teData/L_' + str(multi_path) + '/Rho3/' + 'teNN_H_LS8.mat')['H_ls_save'][:].T
    for snr in SNR:
        Temp_LS = h5py.File('teData/L_' + str(multi_path) + '/Rho3/' + 'teNN_H_LS'+str(snr+8)+'.mat')['H_ls_save'][:].T
        Pre_LSchannel = np.concatenate((Pre_LSchannel,Temp_LS),axis=0)

    # --信道标签
    H_source = h5py.File('teData/L_' + str(multi_path) + '/Rho3/' + 'teNN_TabH8.mat')['H_source_save'][:].T
    for snr in SNR:
        Temp_H = h5py.File('teData/L_' + str(multi_path) + '/Rho3/' + 'teNN_TabH'+str(snr+8)+'.mat')['H_source_save'][:].T
        H_source = np.concatenate((H_source,Temp_H),axis=0)

    # --接收信号
    Rx_data = h5py.File('teData/L_' + str(multi_path) + '/Rho3/' + 'teNN_Rxdata8.mat')['Y_save'][:].T    # --接收到的信号
    for snr in SNR:
        Temp_Rx = h5py.File('teData/L_' + str(multi_path) + '/Rho3/' + 'teNN_Rxdata'+str(snr+8)+'.mat')['Y_save'][:].T
        Rx_data = np.concatenate((Rx_data,Temp_Rx),axis=0)

    # --接收信号
    Tab_data = h5py.File('teData/L_' + str(multi_path) + '/Rho3/' + 'teNN_Tabdata8.mat')['s_save'][:].T # --发送的QPSK信号
    for snr in SNR:
        Temp_tab = h5py.File('teData/L_' + str(multi_path) + '/Rho3/' + 'teNN_Tabdata'+str(snr+8)+'.mat')['s_save'][:].T
        Tab_data = np.concatenate((Tab_data,Temp_tab),axis=0)


    #---经过LS预估计的信道分成实部虚部并拼接
    h_ls = np.zeros((len(Pre_LSchannel),carrier_num,len(Pre_LSchannel.T),2))
    h_ls[:,:,:,0] = Pre_LSchannel['real']
    h_ls[:,:,:,1] = Pre_LSchannel['imag']
    h_ls = np.concatenate((h_ls[:, :, :,0], h_ls[:, :,:, 1]),axis=0).reshape(2*len(Pre_LSchannel), carrier_num, len(Pre_LSchannel.T), 1)

    # ---原始信道的信道分成实部虚部并拼接
    h_tab = np.zeros((len(H_source),carrier_num,len(H_source.T),2))
    h_tab[:,:,:,0] = H_source['real']
    h_tab[:,:,:,1] = H_source['imag']
    h_tab = np.concatenate((h_tab[:,:,:,0] , h_tab[:,:,:,1] ), axis=0).reshape(2*len(H_source), carrier_num, len(H_source.T), 1)
    h_tab = h_tab.reshape((2*len(H_source), len(H_source.T)* 256,1), order='F')
    h_tab = h_tab.transpose(0,2,1)
    # print(perfect[0:5, 0:5])
    # ---发送的经过调制后的信号分成实部虚部并拼接
    s_tab = np.zeros((len(Tab_data),carrier_num,len(Tab_data.T),2))
    s_tab[:,:,:,0] = Tab_data['real']
    s_tab[:,:,:,1] = Tab_data['imag']
    s_tab = np.concatenate((s_tab[:,:,:,0], s_tab[:,:,:,1]), axis=0).reshape(2*len(Pre_LSchannel), carrier_num, len(Pre_LSchannel.T), 1)

    # ---训练集，校验集以及测试集的划分
    data_tr, data_check, label_tr, label_check = train_test_split(h_ls, h_tab, test_size=0.4)
    data_val, data_te, label_val, label_te = train_test_split(data_check, label_check, test_size=0.5)

    return data_tr, label_tr, data_val, label_val, data_te, label_te


def load_Testcsi_datesets(SNR, sub_num, multi_path):

    input_noisy = h5py.File('teData/L_' + str(multi_path) + '/Rho3/' + 'teNN_H_LS'+str(SNR)+'.mat')['H_ls_save'][:].T
    input_mmse = h5py.File('teData/L_' + str(multi_path) + '/Rho3/' + 'teNN_H_MMSE'+str(SNR)+'.mat')['H_mmse_save'][:].T
    perfect = h5py.File('teData/L_' + str(multi_path) + '/Rho3/' + 'teNN_TabH'+str(SNR)+'.mat')['H_source_save'][:].T


    zf_data = h5py.File('teData/L_' + str(multi_path) + '/Rho3/' + 'teNN_LS_Equali_ZF' + str(SNR) + '.mat')['NN_LS_Equali_ZF_save'][:].T
    zfzf_data = h5py.File('teData/L_' + str(multi_path) + '/Rho3/' + 'teNN_LS_Equali_MMSE' + str(SNR) + '.mat')['NN_LS_Equali_MMSE_save'][:].T
    mmse_data = h5py.File('teData/L_' + str(multi_path) + '/Rho3/'  + 'teNN_MMSE_Equali_ZF' + str(SNR) + '.mat')['NN_MMSE_Equali_ZF_save'][:].T
    mmse_mmse_data = h5py.File('teData/L_' + str(multi_path) + '/Rho3/'  + 'teNN_MMSE_Equali_MMSE' + str(SNR) + '.mat')['NN_MMSE_Equali_MMSE_save'][:].T

    rx_data = h5py.File('teData/L_' + str(multi_path) + '/Rho3/'  + 'teNN_Rxdata'+str(SNR)+'.mat')['Y_save'][:].T    # --接收到的信号
    tab_data = h5py.File('teData/L_' + str(multi_path) + '/Rho3/'  + 'teNN_Tabdata' + str(SNR) + '.mat')['s_save'][:].T


    #---经过LS预估计的信道分成实部虚部并拼接
    h_ls = np.zeros((len(input_noisy),sub_num,len(input_noisy.T),2))
    h_ls[:,:,:,0] = input_noisy['real']
    h_ls[:,:,:,1] = input_noisy['imag']
    h_ls = np.concatenate((h_ls[:, :, :,0], h_ls[:, :,:, 1]),axis=0).reshape(2*len(input_noisy), sub_num, len(input_noisy.T), 1)

    #---经过MMSE预估计的信道分成实部虚部并拼接
    h_mmse = np.zeros((len(input_mmse),sub_num,len(input_mmse.T),2))
    h_mmse[:,:,:,0] = input_mmse['real']
    h_mmse[:,:,:,1] = input_mmse['imag']
    h_mmse = np.concatenate((h_mmse[:, :, :,0], h_mmse[:, :,:, 1]),axis=0).reshape(2*len(input_mmse), sub_num, len(input_mmse.T), 1)


    # ---原始信道的信道分成实部虚部并拼接
    h_tab = np.zeros((len(perfect),sub_num,len(perfect.T),2))
    h_tab[:,:,:,0]  = perfect['real']
    h_tab[:,:,:,1]  = perfect['imag']
    h_tab = np.concatenate((h_tab[:,:,:,0] , h_tab[:,:,:,1] ), axis=0).reshape(2*len(perfect), sub_num, len(perfect.T), 1)

    return h_ls, h_mmse, h_tab, zf_data, zfzf_data,mmse_data, mmse_mmse_data,rx_data, tab_data,input_mmse



def load_Testsymbol_datesets_v7(SNR, sub_num):


    input_H_LS = h5py.File('teData/L_8/P_8/' + 'teNN_H_LS'+str(SNR)+'.mat')['H_ls_save'][:].T
    h_lsTe = split_RealImage_3Dim(len(input_H_LS), sub_num, len(input_H_LS.T), input_H_LS)     # --训练集的实虚部拼接

    ZF_data = h5py.File('teData/L_8/P_8/' + 'teNN_LS_Equali_ZF' + str(SNR) + '.mat')['NN_LS_Equali_ZF_save'][:].T
    MMSE_data = h5py.File('teData/L_8/P_8/' + 'teNN_MMSE_Equali_ZF' + str(SNR) + '.mat')['NN_MMSE_Equali_ZF_save'][:].T

    Rx_data = h5py.File('teData/L_8/P_8/' + 'teNN_Rxdata'+str(SNR)+'.mat')['Y_save'][:].T    # --接收到的信号
    Tab_data = h5py.File('teData/L_8/P_8/' + 'teNN_Tabdata' + str(SNR) + '.mat')['s_save'][:].T


    return h_lsTe, Rx_data, Tab_data, ZF_data, MMSE_data



def load_Testsymbol_datesets(SNR, sub_num):


    input_H_LS = loadmat('teData/L_8/P_8/' + 'teNN_H_LS'+str(SNR)+'.mat')['H_ls_save']
    h_lsTe = np.reshape(input_H_LS, (sub_num, 256, int(len(input_H_LS.T) / 256)), order='F').transpose(2, 0, 1)

    #---经过LS预估计的信道分成实部虚部并拼接
    h_ls = np.zeros((len(h_lsTe),sub_num,len(h_lsTe.T), 2))
    h_ls[:,:,:,0] = np.real(h_lsTe)
    h_ls[:,:,:,1] = np.imag(h_lsTe)
    h_ls = np.concatenate((h_ls[:, :, :,0], h_ls[:, :,:, 1]),axis=0).reshape(2*len(h_lsTe), sub_num, len(h_lsTe.T), 1)

    ZF_data = loadmat('teData/L_8/P_8/' + 'teNN_LS_Equali_ZF' + str(SNR) + '.mat')['NN_LS_Equali_ZF_save']
    MMSE_data = loadmat('teData/L_8/P_8/' + 'teNN_MMSE_Equali_ZF' + str(SNR) + '.mat')['NN_MMSE_Equali_ZF_save']

    Rx_data2dim = loadmat('teData/L_8/P_8/' + 'teNN_Rxdata'+str(SNR)+'.mat')['Y_save']
    Rxdata3dim = np.reshape(Rx_data2dim, (sub_num, 256, int(len(Rx_data2dim.T) / 256)), order='F').transpose(2, 0, 1)

    Tab_data = loadmat('teData/L_8/P_8/' + 'teNN_Tabdata' + str(SNR) + '.mat')['s_save']

    tah_H = loadmat('teData/L_8/P_8/' + 'teNN_TabH'+str(SNR)+'.mat')['H_source_save']
    perfect = np.reshape(tah_H, (sub_num, 256, int(len(tah_H.T) / 256)), order='F').transpose(2, 0, 1)
    return h_ls, Rx_data2dim, Rxdata3dim,Tab_data, ZF_data, MMSE_data, perfect,tah_H


# 计算NMSE
def calc_nmse(ce_h,h_source):

    h_source_norm = h_source / (np.linalg.norm(h_source,ord = 2,axis=0))
    ce_h_norm = ce_h / (np.linalg.norm(ce_h,ord= 2,axis= 0))

    temp = np.linalg.norm((ce_h_norm - h_source_norm), ord= 2 ,axis= 0 ) ** 2
    NMSE = np.mean(temp / (np.linalg.norm(h_source_norm,ord=2,axis=0) ** 2))

    return NMSE


# 傅里叶矩阵，J，I
def fourier_matrix(N, P, SNR):
    Q = int(N / P)
    w = np.exp(-1j * 2 * (np.pi) / N)
    temp = np.ones((N, N))
    b = np.argwhere(temp == 1)
    Fourier = w ** (b[:, 0] * b[:, 1])
    Fourier = 1 / (np.sqrt(N)) * np.reshape(Fourier, [N, N])

    Maxtri_1 = np.ones((Q, Q))
    I_p = np.eye(P)
    K = np.kron(Maxtri_1, I_p)
    J = (1 / Q) * K
    I = np.eye(N)

    tatol_power = 1.0
    Noise_power = tatol_power / (np.power(10, np.multiply(0.1, SNR)))



    return Fourier, J, I, Q, Noise_power




# 迫零检测
def signal_detect(Rx_data, H_hat, SNR, P, N):

    Rx_data = np.squeeze(Rx_data)
    a1 = Rx_data['real']
    a2 = Rx_data['imag']
    Rx_dataTem = a1 + 1j * a2
    H_hat = np.squeeze(H_hat)



    Walsh = h5py.File('Walsh.mat')['W'][:].T
    # F = h5py.File('F.mat')['F'][:]
    # b1 = F['real']
    # b2 = F['imag']
    # Fourier = b1 + 1j * b2
    J = h5py.File('J.mat')['J'][:].T

    I = np.eye(N)
    # Fourier, J, I, Q, Noise_power = fourier_matrix(N=N, P=P, SNR=SNR)  # 傅里叶变换矩阵
    Temp = np.matmul((I - J),Rx_dataTem)  # 去除叠加训练序列

    # ZF均衡
    H3 = 1./H_hat
    H3_temp = H3*Temp
    temp2 = np.transpose(np.conj(Walsh))
    u_zf = np.matmul(temp2,H3_temp)  # 均衡

    # MMSE均衡
    Noise_power = 1 / (np.power(10, np.multiply(0.1, SNR)))
    G_mmse = np.conj(H_hat) / (np.power(np.abs(H_hat), 2) + Noise_power)
    H4_temp = G_mmse*Temp
    u_mmse = np.matmul(temp2,H4_temp)


    return u_zf,u_mmse




# QPSK解调
def data_Map(x):

    temp = x
    temp = np.reshape(temp,(1,np.size(temp)))
    temp[:, (np.argwhere(temp > 0))[:, 1]] = 1
    temp[:, (np.argwhere(temp <=  0))[:, 1]] = 0

    return temp

# 计算BER函数
def BER_Calc(x,y):
    x_temp = data_Map(x)
    y_temp = data_Map(y)

    num_x = np.size(x_temp)
    temp = np.sum(np.abs((x_temp - y_temp)))
    Ber = (temp)/num_x

    return Ber


# 符号检测数据处理
def SD_detesets_Processing( sub_num, input_X, input_Y, Label):


    rx_data = np.zeros((sub_num, len(input_Y.T), 2))
    rx_data[:, :, 0] = np.real(input_Y)
    rx_data[:, :, 1] = np.imag(input_Y)
    rx_data = np.concatenate((rx_data[:, :, 0], rx_data[:, :, 1]), axis=0)  # 实虚部分开

    channeANDy = np.hstack((input_X, rx_data.T))
    #  得到符号检测的训练数据
    sig_tr, sig_data_check, sig_label_tr, sig_label_check = train_test_split(channeANDy, Label.T, test_size=0.4)
    sig_val, sig_te, sig_label_val, sig_label_te = train_test_split(sig_data_check, sig_label_check,
                                                                              test_size=0.2)

    return sig_tr, sig_label_tr, sig_val, sig_label_val, sig_te, sig_label_te




# import deepdish as dd
# file_path1 =  'result/06-23-08-50-21_rr_0.0001lr_0.0001/Transfer_0/Trans_nn.h5'
#
# def load_h5(file_path1):
#     mean_val1 = dd.io.load(file_path1)
#     print(mean_val1)
#
# load_h5(file_path1)