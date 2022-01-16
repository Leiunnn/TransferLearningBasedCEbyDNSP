from utils import BER_Calc, load_Testcsi_datesets, calc_nmse, split_RealImage_2Dim,load_Testsymbol_datesets, signal_detect
from models import SRCNN_predict,DNCNN_predict,TrainTransfer_net,PrediTransfer_net,TransferCNN_predict
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

def TrainLoss_Plot(tr_loss, val_loss ,path):

    # 训练集损失图
    loss_history_train = np.array(tr_loss)
    plt.plot(loss_history_train)
    plt.xlabel('Iters')
    plt.ylabel('loss')
    plt.title('Train loss')
    plt.savefig(path + '/train_loss.png')
    plt.close()


    # 校验集集损失图
    val_loss_file = path + 'val_loss.txt'
    np.savetxt(val_loss_file, val_loss, delimiter=',')
    loss_history_val = np.array(val_loss)
    plt.plot(loss_history_val)
    plt.xlabel('Iters')
    plt.ylabel('loss')
    plt.title('Val loss')
    plt.savefig(path + '/val_loss.png')
    plt.close()

# ======绘制信道估计的测试曲线=======
def TestCE_dataPlot(SNR, sub_num, path, path4, multi_path, channel_model, phase):
    LS_MSE = []
    MMSE_MSE = []
    Temp_nmse1 = []

    Temp_Ber = []
    TempZF_Ber = []
    TempMMSE_Ber = []
    TempMMSE_BerBER = []
    TempMMSE_MMSEBerBER = []
    TempMMSE_dae_aTeBER = []

    for snr in SNR:

        INPUT_H_ls, INPUT_H_mmse, H_source, ZF_data, ZFzf_data, MMSE_data, MMSEmmse_data,RX_data, TAB_data, input_mmse = load_Testcsi_datesets(snr, sub_num,  multi_path)

        TransPredTe = TransferCNN_predict(INPUT_H_ls, channel_model, path4, False)

        # --2维变成3维
        pre_h = TransPredTe.reshape((len(H_source), 256, 16,1), order='F')

        # --算NMSE
        LS_MSEpy = calc_nmse(INPUT_H_ls, H_source)
        MMSE_MSEpy = calc_nmse(INPUT_H_mmse, H_source)
        Test_NMSE1 = calc_nmse(pre_h, H_source)

        # --实数变成复数
        H_tempTe = pre_h[0:int(0.5 * len(pre_h)), :, :, :] + 1j * pre_h[int(0.5 * len(pre_h)):len(pre_h), :, :, :]
        Hmmse_tempTe = INPUT_H_mmse[0:int(0.5 * len(INPUT_H_mmse)), :, :, :] + 1j * INPUT_H_mmse[int(0.5 * len(INPUT_H_mmse)):len(INPUT_H_mmse), :, :, :]

        a1 = input_mmse['real']
        a2 = input_mmse['imag']
        INPUT_H_MMMSE = a1 + 1j * a2

        # --做ZF均衡
        u_zfTe,u_mmseTe = signal_detect(RX_data, H_tempTe, snr, P=8, N=sub_num)
        ummse_zfTe,ummse_mmseTe = signal_detect(RX_data, INPUT_H_MMMSE, snr, P=8, N=sub_num)

        # --均衡后的数据拼接为2维
        u_zf1_te = np.squeeze(u_zfTe[0, :, :])
        for iii in range(len(u_zfTe) - 1):
            u_zf1_te = np.concatenate((np.squeeze(u_zfTe[iii + 1, :, :]), u_zf1_te), axis=1)


        u_tab_te = np.squeeze(TAB_data[0, :, :])
        for iii in range(len(TAB_data) - 1):
            u_tab_te = np.concatenate((np.squeeze(TAB_data[iii + 1, :, :]), u_tab_te), axis=1)

        zf_data_te = np.squeeze(ZF_data[0, :, :])
        for iii in range(len(ZF_data) - 1):
            zf_data_te = np.concatenate((np.squeeze(ZF_data[iii + 1, :, :]), zf_data_te), axis=1)


        mmse_data_te = np.squeeze(ummse_zfTe[0, :, :])
        for iii in range(len(ummse_zfTe) - 1):
            mmse_data_te = np.concatenate((np.squeeze(ummse_zfTe[iii + 1, :, :]), mmse_data_te), axis=1)

        MMSE_dae = np.squeeze(ummse_mmseTe[0, :, :])
        for iii in range(len(ummse_mmseTe) - 1):
            MMSE_dae = np.concatenate((np.squeeze(ummse_mmseTe[iii + 1, :, :]), MMSE_dae), axis=1)

        mmse_hhh_te = np.squeeze(ZFzf_data[0, :, :])
        for iii in range(len(ZFzf_data) - 1):
            mmse_hhh_te = np.concatenate((np.squeeze(ZFzf_data[iii + 1, :, :]), mmse_hhh_te), axis=1)

        mmse_mmse_te = np.squeeze(u_mmseTe[0, :, :])
        for iii in range(len(u_mmseTe) - 1):
            mmse_mmse_te = np.concatenate((np.squeeze(u_mmseTe[iii + 1, :, :]), mmse_mmse_te), axis=1)

        # ---分成实部与虚部
        s_zfTe = split_RealImage_2Dim(sub_num, len(u_zf1_te.T),u_zf1_te,flag=None)

        smmse_zfTe = split_RealImage_2Dim(sub_num, len(mmse_hhh_te.T),mmse_hhh_te,flag='v7')

        s_tabTe = split_RealImage_2Dim(sub_num, len(u_tab_te.T),u_tab_te,flag='v7')

        zf_dataTe = split_RealImage_2Dim(sub_num, len(zf_data_te.T),zf_data_te,flag='v7')

        mmse_dataTe = split_RealImage_2Dim(sub_num, len(mmse_data_te.T),mmse_data_te,flag=None)

        mmseMMSE_dataTe = split_RealImage_2Dim(sub_num, len(mmse_mmse_te.T),mmse_mmse_te,flag=None)
        MMSE_dae_aTe = split_RealImage_2Dim(sub_num, len(MMSE_dae.T),MMSE_dae,flag=None)

        # --算BER
        ber1 = BER_Calc(s_zfTe, s_tabTe)                                    # --计算误码率
        ber2 = BER_Calc(zf_dataTe, s_tabTe)                                 # --计算误码率
        ber3 = BER_Calc(mmse_dataTe, s_tabTe)                               # --计算误码率
        ber4 = BER_Calc(smmse_zfTe, s_tabTe)                               # --计算误码率
        ber5 = BER_Calc(mmseMMSE_dataTe, s_tabTe)                               # --计算误码率
        ber6 = BER_Calc(MMSE_dae_aTe, s_tabTe)                               # --计算误码率


        LS_MSE.append(LS_MSEpy)
        MMSE_MSE.append(MMSE_MSEpy)
        Temp_nmse1.append(Test_NMSE1)

        Temp_Ber.append(ber1)
        TempZF_Ber.append(ber2)
        TempMMSE_Ber.append(ber3)
        TempMMSE_BerBER.append(ber4)
        TempMMSE_MMSEBerBER.append(ber5)
        TempMMSE_dae_aTeBER.append(ber6)

    np.savetxt(path4+'LS_MSE.txt', LS_MSE, delimiter=',')
    np.savetxt(path4+'MMSE_MSE.txt', MMSE_MSE, delimiter=',')
    np.savetxt(path4+'DL_MSE.txt', Temp_nmse1, delimiter=',')
    np.savetxt(path4+'DL_BER.txt', Temp_Ber, delimiter=',')
    np.savetxt(path4+'LS_BER.txt', TempZF_Ber, delimiter=',')
    np.savetxt(path4+'MMSE_BER.txt', TempMMSE_Ber, delimiter=',')
    np.savetxt(path4+'LS_MMSE_BER.txt', TempMMSE_BerBER, delimiter=',')
    np.savetxt(path4+'MMSE_MMSE_BER.txt', TempMMSE_dae_aTeBER, delimiter=',')
    np.savetxt(path4+'DL_MMSE_BER.txt', TempMMSE_MMSEBerBER, delimiter=',')
    # 测试经过网络后的NMSE图
    plt.figure()
    plt.semilogy(SNR, LS_MSE, color="blue", linewidth=1.0, linestyle="--", marker='s', label='LS')
    plt.semilogy(SNR, MMSE_MSE, color="green", linewidth=1.0, linestyle="--", marker='>', label='MMSE')
    plt.semilogy(SNR, Temp_nmse1, color="red", linewidth=1.0, linestyle="-", marker='*', label='Transfer')
    plt.grid()
    plt.xlabel('SNR')
    plt.ylabel('NMSE')
    plt.legend()
    plt.title('Deep Learning on channel estimation')
    plt.savefig(path + '/Test_for_NMSE_'+ phase +'.pdf')
    plt.close()

    print('Begin to Training SD-Net')


    # 测试经过网络后的NMSE图
    plt.figure()
    plt.semilogy(SNR, Temp_Ber, color="red", linewidth=1.0, linestyle="-", marker='+', label='DL')
    plt.semilogy(SNR, TempZF_Ber, color="blue", linewidth=1.0, linestyle="--", marker='o', label='LS+ZF')
    plt.semilogy(SNR, TempMMSE_Ber, color="green", linewidth=1.0, linestyle="--", marker='s', label='MMSE+ZF')
    plt.semilogy(SNR, TempMMSE_BerBER, color="black", linewidth=1.0, linestyle="--", marker='>', label='LS+MMSE')
    plt.semilogy(SNR, TempMMSE_MMSEBerBER, color="purple", linewidth=1.0, linestyle="--", marker='>', label='DL+MMSE')
    plt.semilogy(SNR, TempMMSE_dae_aTeBER, color="yellow", linewidth=1.0, linestyle="--", marker='<', label='MMSE+MMSE')
    plt.grid()
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.legend()
    plt.title('Deep Learning on symbol detection')
    plt.savefig(path + '/Test_for_BER_'+ phase +'.pdf')
    plt.close()

    print('One Training Over')




# ======绘制符号检测的测试曲线=======
def TestSD_NET_dataPlot(SNR, sub_num, path, path1, path2, rr, lr, channel_model):

    Temp_Ber = []
    TempZF_Ber = []
    TempMMSE_Ber = []
    TempTran_Ber = []
    for snr in SNR:
        print('One SNR_'+str(snr))
        path3 = 'result/06-23-08-50-21_rr_0.0001lr_0.0001/Transfer_'+ str(snr)+'/'

        h_ls, Rx_data2dim, Rxdata3dim, s_tab, zf_data, mmse_data, perfect,tah_H_2dim = load_Testsymbol_datesets(snr, sub_num)   # --读取符号检测所用测试数据

        # --经过CE-Net得到估计的CSI
        srcnn_s = SRCNN_predict(h_ls, channel_model, path1)
        sdnn_s = DNCNN_predict(srcnn_s, channel_model, path2)


        # --经过Direct-Net得到估计的CSI--->将三维的图片变化为2维的方阵
        H_tempTr = np.squeeze(sdnn_s[0:int(0.5 * len(sdnn_s)), :, :, :] + 1j * sdnn_s[int(0.5 * len(sdnn_s)):len(sdnn_s), :, :, :])
        TranCnnOut = H_tempTr.transpose(1, 2, 0).reshape((sub_num, len(H_tempTr) * 256), order='F')

        h_cnn = split_RealImage_2Dim(sub_num, len(TranCnnOut.T), TranCnnOut, flag=None)

        # --送到微调层
        tran_hat = PrediTransfer_net( h_cnn.T, rr, lr, path3)


        # ---先变成复数CSI
        H_tempTe = sdnn_s[0:int(0.5 * len(sdnn_s)), :, :, :] + 1j * sdnn_s[int(0.5 * len(sdnn_s)):len(sdnn_s),:, :, :]

        tran_h = tran_hat.T
        H_tempTras = tran_h[0:int(0.5 * len(tran_h)), :] + 1j * tran_h[int(0.5 * len(tran_h)):len(tran_h),:]


        # ---做ZF均衡
        u_zfTe = signal_detect(Rxdata3dim, perfect, snr, P=8, N=sub_num)


        u_zfTran = signal_detect(Rx_data2dim, tah_H_2dim, snr, P=8, N=sub_num)

        # ---Reshape为2维形状
        u_zf1_te = np.squeeze(u_zfTe[0, :, :])
        for iii in range(len(u_zfTe) - 1):
            u_zf1_te = np.concatenate((np.squeeze(u_zfTe[iii + 1, :, :]), u_zf1_te), axis=1)  # --SD-Net的训练输入

        u_tab_te = np.squeeze(s_tab[0, :, :])
        for iii in range(len(s_tab) - 1):
            u_tab_te = np.concatenate((np.squeeze(s_tab[iii + 1, :, :]), u_tab_te), axis=1)  # --SD-Net的训练输入

        zf_data_te = np.squeeze(zf_data[0, :, :])
        for iii in range(len(zf_data) - 1):
            zf_data_te = np.concatenate((np.squeeze(zf_data[iii + 1, :, :]), zf_data_te), axis=1)  # --SD-Net的训练输入

        mmse_data_te = np.squeeze(mmse_data[0, :, :])
        for iii in range(len(mmse_data) - 1):
            mmse_data_te = np.concatenate((np.squeeze(mmse_data[iii + 1, :, :]), mmse_data_te), axis=1)  # --SD-Net的训练输入

        # ---分成实部与虚部
        s_zfTe = split_RealImage_2Dim(sub_num, len(u_zf1_te.T),u_zf1_te,flag=None)

        s_tabTe = split_RealImage_2Dim(sub_num, len(u_tab_te.T),u_tab_te,flag=None)

        zf_dataTe = split_RealImage_2Dim(sub_num, len(zf_data_te.T),zf_data_te,flag=None)

        mmse_dataTe = split_RealImage_2Dim(sub_num, len(mmse_data_te.T),mmse_data_te,flag=None)

        s_zfTran = split_RealImage_2Dim(sub_num, len(u_zfTran.T),u_zfTran,flag=None)

        ber1 = BER_Calc(s_zfTe, s_tabTe)                                    # --计算误码率
        ber2 = BER_Calc(zf_dataTe, s_tabTe)                                 # --计算误码率
        ber3 = BER_Calc(mmse_dataTe, s_tabTe)                               # --计算误码率
        ber4 = BER_Calc(s_zfTran, s_tabTe)                                  # --计算误码率

        Temp_Ber.append(ber1)
        TempZF_Ber.append(ber2)
        TempMMSE_Ber.append(ber3)
        TempTran_Ber.append(ber4)

    # 测试经过网络后的NMSE图
    plt.figure()
    plt.semilogy(SNR, Temp_Ber, color="red", linewidth=1.0, linestyle="-", marker='+', label='DL')
    plt.semilogy(SNR, TempZF_Ber, color="blue", linewidth=1.0, linestyle="--", marker='o', label='LS+ZF')
    plt.semilogy(SNR, TempMMSE_Ber, color="black", linewidth=1.0, linestyle="--", marker='s', label='MMSE+ZF')
    plt.semilogy(SNR, TempTran_Ber, color="purple", linewidth=1.0, linestyle="-", marker='s', label='TransDirect')
    plt.grid()
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.legend()
    plt.title('Deep Learning on symbol detection')
    plt.savefig(path + '/Test_for_BER.pdf')
    plt.close()

    print('One Training Over')



# ======绘制迁移网络的测试曲线=======
def Direct_Net_dataPlot(SNR, sub_num, D_in, hidden, D_out, path1, path2, rr, lr, multi_path):
    LS_method = [0.581063083489278,0.253023191847235,0.0901513343234237,0.0300296682114491,0.00963759396319783,0.00305844322900611]
    MMSE_method = [0.0336954775410413,0.0106434759389424,0.00317697325930596,0.00103770747599212,0.000320446844252514,0.000101350671671148]
    Temp_nmse = []
    for snr in SNR:

        INPUT_H_ls, H_source = load_Testcsi_datesets(snr, sub_num,  multi_path)

        H_hat = CE_Dnn_predict(te_data=INPUT_H_ls,
                               D_in=D_in,
                               hidden=hidden,
                               D_out=D_out,
                               path1=path1,
                               rr=rr,
                               lr=lr)

        TranH_hat = PrediTransfer_net(te_data=H_hat,
                                      InPut_Node=D_in,
                                      OutPut_Node=D_out,
                                      rr=rr,
                                      lr=lr,
                                      path2=path2)

        Test_NMSE = calc_nmse(TranH_hat.T, H_source.T)

        Temp_nmse.append(Test_NMSE)


    # 测试经过网络后的NMSE图
    plt.figure()
    plt.semilogy(SNR, LS_method, color="black", linewidth=1.0, linestyle="--", marker='o', label='LS_method')
    plt.semilogy(SNR, MMSE_method, color="blue", linewidth=1.0, linestyle="--", marker='v', label='MMSE_method')
    plt.semilogy(SNR, Temp_nmse, color="red", linewidth=1.0, linestyle="-", marker='+', label='DL')
    plt.grid()
    plt.xlabel('SNR')
    plt.ylabel('NMSE')
    plt.legend()
    plt.title('Deep Learning on channel estimation')
    plt.savefig(path2 + '/Test_for_NMSE.png')
    plt.close()











