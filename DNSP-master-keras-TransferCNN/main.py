from models import SRCNN_train,SRCNN_predict,DNCNN_predict,DNCNN_train,TransferCNN_train,TransferCNN_finetuning,TransferCNN_predict
from utils import load_Trdatasets, calc_nmse, SD_detesets_Processing, load_Testcsi_datesets,load_Finedatasets
from plot_figure import TrainLoss_Plot, TestCE_dataPlot, TestSD_NET_dataPlot
from sklearn.model_selection import train_test_split
import os
import numpy as np
import time
import argparse

# --GPU切换
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# --参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--carrier_num', type=int, default=256, help='The carrier num.')
parser.add_argument('--D_in', type=int, default=256*2, help='Input net node number for CE-Net.')
parser.add_argument('--D_inSD', type=int, default=256*4, help='Input net node number for SD-Net.')
parser.add_argument('--Hidden', type=int, default=256*2, help='Hidden net node number.')
parser.add_argument('--D_out', type=int, default=256*2, help='Output net node number.')
parser.add_argument('--batch_size', type=int, default=80, help='Batch size number.')
parser.add_argument('--epochs', type=int, default=6, help='Training epoch num.')
parser.add_argument('--Flag_tr',type=bool, default=True)
parser.add_argument('--Flag_los',type=bool,default=True)
args = parser.parse_args()
# -- 信道模型
channel_model = "COST2100"
# -- 仿真信噪比
SNR = [0,2,4,6,8,10,12,14,16,18,20]

# -- 离线训练多径数
multi_path = 88

# -- 训练信噪比
Tr_SNR = [8,16,24]

# --学习率与正则化系数
regular_rate = 10**(-np.linspace(4,4,1))  # -4--->-5
learning_rate = 10**(-np.linspace(4,4,1))  # -3--->-4

if __name__ == "__main__":
    for lr in learning_rate:
        for rr in regular_rate:

            path = 'result/' + time.strftime('%m-%d-%H-%M-%S_' + 'rr_' + str(rr) + 'lr_' + str(lr) + '/')   # --结果保存路径
            os.makedirs(path)

            # path = 'result/07-01-15-44-52_rr_0.0001lr_0.0001/'
            path4 = 'result/07-08-16-17-20_rr_0.0001lr_0.0001/TransferCNN_Pre/'

            # --加载数据集
            # data_tr, label_tr, data_val, label_val, data_te, label_te = load_Trdatasets(carrier_num=args.carrier_num,SNR=Tr_SNR)

            # --预训练
            # Pretrain_loss, Preval_loss, path4 = TransferCNN_train(data_tr, label_tr, data_val, label_val, channel_model, path, True)

            # --预训练损失图
            # TrainLoss_Plot(tr_loss=Pretrain_loss, val_loss=Preval_loss, path=path4)

            # --与训练后的测试
            TestCE_dataPlot(SNR, args.carrier_num, path, path4, multi_path, channel_model, 'Pre')

            # --加载微调数据集
            fine_tr, labTr_fine, fine_val, labVal_fine, fine_te, labTe_fine = load_Finedatasets(carrier_num=args.carrier_num,multi_path=multi_path,SNR=Tr_SNR)

            # --微调
            Fintrain_loss, Finval_loss, path5 = TransferCNN_finetuning(fine_tr, labTr_fine, fine_val, labVal_fine, channel_model, path, path4, False)

            # --微调损失图
            TrainLoss_Plot(tr_loss=Fintrain_loss, val_loss=Finval_loss, path=path5)

            # --微调后的测试
            TestCE_dataPlot(SNR, args.carrier_num, path, path5, multi_path, channel_model, 'Fin')



