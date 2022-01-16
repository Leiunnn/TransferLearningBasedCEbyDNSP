from models import SRCNN_train,SRCNN_predict,DNCNN_predict,DNCNN_train
from utils import BER_Calc, calc_nmse, signal_detect
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
# -- 信道模型
channel_model = "COST2100"
# -- 子载波数
sub_num = 256
# -- 仿真信噪比
SNR = [0,3,6,9,12,15,18,21,24,27,30]
# -- 离线训练多径数
multi_path = 8

path = 'result/06-06-10-01-35_rr_0.0001lr_0.0001/'
path1 = 'result/06-06-10-01-35_rr_0.0001lr_0.0001/SRCNN/'
path2 = 'result/06-06-10-01-35_rr_0.0001lr_0.0001/DNCNN/'

def load_cis_datesets(SNR, sub_num, multi_path):

    input_noisy = loadmat('teData/L_' + str(multi_path) + '/P_8/' + 'teNN_H_LS'+str(SNR)+'.mat')['H_ls_save']
    input_MMSE = loadmat('teData/L_' + str(multi_path) + '/P_8/' + 'teNN_H_MMSE'+str(SNR)+'.mat')['H_mmse_save']
    perfect = loadmat('teData/L_' + str(multi_path) + '/P_8/' + 'teNN_TabH'+str(SNR)+'.mat')['H_source_save']
    Rx_data = loadmat('teData/L_' + str(multi_path) + '/P_8/' + 'teNN_Rxdata'+str(SNR)+'.mat')['Y_save']
    Tab_data = loadmat('teData/L_' + str(multi_path) + '/P_8/' + 'teNN_Tabdata'+str(SNR)+'.mat')['s_save']
    ZF_data = loadmat('teData/L_' + str(multi_path) + '/P_8/' + 'teNN_LS_Equali_ZF'+str(SNR)+'.mat')['NN_LS_Equali_ZF_save']
    MMSE_data = loadmat('teData/L_' + str(multi_path) + '/P_8/' + 'teNN_MMSE_Equali_ZF'+str(SNR)+'.mat')['NN_MMSE_Equali_ZF_save']

    #---经过LS预估计的信道分成实部虚部并拼接
    h_ls = np.zeros((len(input_noisy),sub_num,len(input_noisy.T),2))
    h_ls[:,:,:,0] = np.real(input_noisy)
    h_ls[:,:,:,1] = np.imag(input_noisy)
    h_ls = np.concatenate((h_ls[:, :, :,0], h_ls[:, :,:, 1]),axis=0).reshape(2*len(input_noisy), sub_num, len(input_noisy.T), 1)

    #---经过MMSE预估计的信道分成实部虚部并拼接
    h_mmse = np.zeros((len(input_MMSE),sub_num,len(input_MMSE.T),2))
    h_mmse[:,:,:,0] = np.real(input_MMSE)
    h_mmse[:,:,:,1] = np.imag(input_MMSE)
    h_mmse = np.concatenate((h_mmse[:, :, :,0], h_mmse[:, :,:, 1]),axis=0).reshape(2*len(input_noisy), sub_num, len(input_noisy.T), 1)

    # ---原始信道的信道分成实部虚部并拼接
    h_tab = np.zeros((len(perfect),sub_num,len(perfect.T),2))
    h_tab[:,:,:,0]  = np.real(perfect)
    h_tab[:,:,:,1]  = np.imag(perfect)
    h_tab = np.concatenate((h_tab[:,:,:,0] , h_tab[:,:,:,1] ), axis=0).reshape(2*len(perfect), sub_num, len(perfect.T), 1)

    # # ---接收端的接收信号
    # rx_data = np.zeros((len(Rx_data),sub_num,len(Rx_data.T),2))
    # rx_data[:,:,:,0]  = np.real(Rx_data)
    # rx_data[:,:,:,1]  = np.imag(Rx_data)
    # rx_data = np.concatenate((rx_data[:,:,:,0] , rx_data[:,:,:,1] ), axis=0).reshape(2*len(perfect), sub_num, len(perfect.T), 1)
    #
    # # ---发送的调制信号
    # tx_data = np.zeros((len(Tab_data),sub_num,len(Tab_data.T),2))
    # tx_data[:,:,:,0]  = np.real(Tab_data)
    # tx_data[:,:,:,1]  = np.imag(Tab_data)
    # tx_data = np.concatenate((tx_data[:,:,:,0] , tx_data[:,:,:,1] ), axis=0).reshape(2*len(perfect), sub_num, len(perfect.T), 1)
    #
    # # ---经LS估计并随后进行ZF均衡的信号
    # ls_zf = np.zeros((len(ZF_data),sub_num,len(ZF_data.T),2))
    # ls_zf[:,:,:,0]  = np.real(ZF_data)
    # ls_zf[:,:,:,1]  = np.imag(ZF_data)
    # ls_zf = np.concatenate((ls_zf[:,:,:,0] , ls_zf[:,:,:,1] ), axis=0).reshape(2*len(perfect), sub_num, len(perfect.T), 1)
    #
    # # ---经MMSE估计并随后进行ZF均衡的信号
    # mmse_zf = np.zeros((len(MMSE_data),sub_num,len(MMSE_data.T),2))
    # mmse_zf[:,:,:,0]  = np.real(MMSE_data)
    # mmse_zf[:,:,:,1]  = np.imag(MMSE_data)
    # mmse_zf = np.concatenate((mmse_zf[:,:,:,0] , mmse_zf[:,:,:,1] ), axis=0).reshape(2*len(perfect), sub_num, len(perfect.T), 1)



    return h_ls, h_mmse,h_tab, Rx_data, Tab_data, ZF_data, MMSE_data

# ======绘制信道估计的测试曲线=======
def TestCE_MPCs(SNR, sub_num, path, path1, path2, multi_path, channel_model):
    # LS_MSE = [1.19,1.06,0.79,0.6,0.42,0.33,0.25,0.21,0.2,0.1945,0.192]
    # MMSE_MSE = [0.416,0.1952,0.1257,0.09,0.07,0.0525,0.046,0.039,0.038,0.037,0.035]
    # MMSE_ZF_BER = [0.36,0.3,0.225,0.1616,0.106,0.07,0.044,0.03,0.023,0.02045,0.019]
    Temp_nmse1 = []
    Temp_nmse2 = []
    Temp_nmse3 = []
    Temp_nmse4 = []

    ber_snrDL = []
    ber_snrLSpy = []
    ber_snrMMSEpy = []
    ber_snrLS = []
    ber_snrMMSE = []
    for snr in SNR:
        ber_dl = []
        ber_lsPy = []
        ber_mmsePy = []
        ber_ls = []
        ber_mmse = []

        h_ls, h_mmse, h_tab, rx_data, tx_data, ls_zf, mmse_zf = load_cis_datesets(snr, sub_num,  multi_path)

        srcnn_pred_test = SRCNN_predict(h_ls, channel_model, path1)
        dncnn_pred_test = DNCNN_predict(srcnn_pred_test, channel_model, path2)

        Test_NMSE1 = calc_nmse(srcnn_pred_test, h_tab)
        Test_NMSE2 = calc_nmse(dncnn_pred_test, h_tab)
        Pytho_LS = calc_nmse(h_ls, h_tab)
        Pytho_MMSE = calc_nmse(h_mmse, h_tab)



        H_temptemp1 = dncnn_pred_test[0:100,:,:,:] + 1j * dncnn_pred_test[100:(2 * 100),:,:,:]  # 变成复数形式
        H_temptemp2 = h_ls[0:100,:,:,:] + 1j * h_ls[100:(2 * 100),:,:,:]  # 变成复数形式
        H_temptemp3 = h_mmse[0:100,:,:,:] + 1j * h_mmse[100:(2 * 100),:,:,:]  # 变成复数形式

        for iii in range(len(H_temptemp1)):

            rx_temp = np.squeeze(rx_data[iii,:,:])
            tx_temp = np.squeeze(tx_data[iii,:,:])
            ls_zf_temp = np.squeeze(ls_zf[iii,:,:])
            mmse_zf_temp = np.squeeze(mmse_zf[iii,:,:])

            H_zf1 = np.squeeze(H_temptemp1[iii,:,:,:])
            H_zf2= np.squeeze(H_temptemp2[iii,:,:,:])
            H_zf3 = np.squeeze(H_temptemp3[iii,:,:,:])


            u_zf1, Walsh, J = signal_detect(rx_temp, H_zf1, snr, P = 8, N=sub_num)
            u_zf2, Walsh, J = signal_detect(rx_temp, H_zf2, snr, P = 8, N=sub_num)
            u_zf3, Walsh, J = signal_detect(rx_temp, H_zf3, snr, P = 8, N=sub_num)

            zf_temp1 = np.zeros((sub_num, len(u_zf1.T), 2))
            zf_temp1[:, :, 0] = np.real(u_zf1)
            zf_temp1[:, :, 1] = np.imag(u_zf1)
            zf_temp1 = np.concatenate((zf_temp1[:, :, 0], zf_temp1[:, :, 1]), axis=0)  # 实虚部分开

            zf_temp2 = np.zeros((sub_num, len(u_zf2.T), 2))
            zf_temp2[:, :, 0] = np.real(u_zf2)
            zf_temp2[:, :, 1] = np.imag(u_zf2)
            zf_temp2 = np.concatenate((zf_temp2[:, :, 0], zf_temp2[:, :, 1]), axis=0)  # 实虚部分开

            zf_temp3 = np.zeros((sub_num, len(u_zf3.T), 2))
            zf_temp3[:, :, 0] = np.real(u_zf3)
            zf_temp3[:, :, 1] = np.imag(u_zf3)
            zf_temp3 = np.concatenate((zf_temp3[:, :, 0], zf_temp3[:, :, 1]), axis=0)  # 实虚部分开


            tab_temp = np.zeros((sub_num, len(tx_temp.T), 2))
            tab_temp[:, :, 0] = np.real(tx_temp)
            tab_temp[:, :, 1] = np.imag(tx_temp)
            tab_temp = np.concatenate((tab_temp[:, :, 0], tab_temp[:, :, 1]), axis=0)  # 实虚部分开

            lsZFdata_temp = np.zeros((sub_num, len(ls_zf_temp.T), 2))
            lsZFdata_temp[:, :, 0] = np.real(ls_zf_temp)
            lsZFdata_temp[:, :, 1] = np.imag(ls_zf_temp)
            lsZFdata_temp = np.concatenate((lsZFdata_temp[:, :, 0], lsZFdata_temp[:, :, 1]), axis=0)  # 实虚部分开

            mmseZFdata_temp = np.zeros((sub_num, len(mmse_zf_temp.T), 2))
            mmseZFdata_temp[:, :, 0] = np.real(mmse_zf_temp)
            mmseZFdata_temp[:, :, 1] = np.imag(mmse_zf_temp)
            mmseZFdata_temp = np.concatenate((mmseZFdata_temp[:, :, 0], mmseZFdata_temp[:, :, 1]), axis=0)  # 实虚部分开


            ber_11 = BER_Calc(zf_temp1, tab_temp)
            ber_12 = BER_Calc(zf_temp2, tab_temp)
            ber_13 = BER_Calc(zf_temp3, tab_temp)
            ber_2 = BER_Calc(lsZFdata_temp, tab_temp)
            ber_3 = BER_Calc(mmseZFdata_temp, tab_temp)

            ber_dl.append(ber_11)
            ber_lsPy.append(ber_12)
            ber_mmsePy.append(ber_13)
            ber_ls.append(ber_2)
            ber_mmse.append(ber_3)

        ber_snrDL.append(np.mean(ber_dl))
        ber_snrLSpy.append(np.mean(ber_lsPy))
        ber_snrMMSEpy.append(np.mean(ber_mmsePy))
        ber_snrLS.append(np.mean(ber_ls))
        ber_snrMMSE.append(np.mean(ber_mmse))

        Temp_nmse1.append(Test_NMSE1)
        Temp_nmse2.append(Test_NMSE2)
        Temp_nmse3.append(Pytho_LS)
        Temp_nmse4.append(Pytho_MMSE)



    # 测试经过网络后的NMSE图
    plt.figure()
    # plt.semilogy(SNR, LS_MSE, color="blue", linewidth=1.0, linestyle="--", marker='s', label='LS')
    # plt.semilogy(SNR, MMSE_MSE, color="green", linewidth=1.0, linestyle="--", marker='>', label='MMSE')
    plt.semilogy(SNR, Temp_nmse1, color="black", linewidth=1.0, linestyle="-", marker='o', label='SRCNN')
    plt.semilogy(SNR, Temp_nmse2, color="red", linewidth=1.0, linestyle="-", marker='+', label='DNCNN')
    plt.semilogy(SNR, Temp_nmse3, color="blue", linewidth=1.0, linestyle="-", marker='+', label='Python_LS')
    plt.semilogy(SNR, Temp_nmse4, color="green", linewidth=1.0, linestyle="-", marker='+', label='Python_MMSE')
    plt.grid()
    plt.xlabel('SNR')
    plt.ylabel('NMSE')
    plt.legend()
    plt.title('Deep Learning on channel estimation')
    # plt.savefig(path + '/Test_for_NMSE.png')
    # plt.close()
    plt.show()



    # 测试经过网络后的NMSE图
    plt.figure()
    plt.semilogy(SNR, ber_snrLS, color="blue", linewidth=1.0, linestyle="--", marker='>', label='LS_ZF')
    plt.semilogy(SNR, ber_snrMMSE, color="black", linewidth=1.0, linestyle="-", marker='o', label='MMSE_ZF')
    plt.semilogy(SNR, ber_snrDL, color="red", linewidth=1.0, linestyle="-", marker='+', label='DNCNN')
    plt.semilogy(SNR, ber_snrLSpy, color="green", linewidth=1.0, linestyle="-", marker='+', label='LS_ZFpyth')
    plt.semilogy(SNR, ber_snrMMSEpy, color="purple", linewidth=1.0, linestyle="-", marker='+', label='MMSE_ZFpyth')
    plt.grid()
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.legend()
    plt.title('Deep Learning on data detection')
    # plt.savefig(path + '/Test_for_NMSE.png')
    # plt.close()
    plt.show()
    print('END')

if __name__ == "__main__":
    TestCE_MPCs(SNR, sub_num, path, path1, path2, multi_path, channel_model)