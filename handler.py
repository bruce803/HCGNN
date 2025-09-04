import json
from datetime import datetime

from data_loader.forecast_dataloader import ForecastDataset, de_normalized
from models.base_model import Model
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.math_utils import evaluate
from utils.math_utils import MA
from utils.math_utils import compute_ext
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc,roc_curve
from ruptures.metrics import precision_recall
import ruptures as rpt
plt.rcParams['font.sans-serif']=['fangsong']
import sys

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def reset(self):
        self.log.close()
        sys.stdout = self.terminal

def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)

def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model

def label(data,window_size):
    if torch.is_tensor(data):
        b,t,N=data.size()
        data=data.view(1,-1,76).squeeze()
        inc=torch.zeros(data.shape)
        for i in range(1,len(data)):
            inc[i,:]=data[i,:]-data[i-1,:]
            label=torch.where(inc > 0, 1, torch.where(inc < 0, -1, 0))
        label=label.view(b, t, N )
        label=torch.tensor(label,dtype=torch.float)
        return label
    elif type(data) == np.ndarray:
        if data.ndim==2:
            t, N = data.shape
        elif data.ndim==3:
            b,t,N = data.shape
            data=data.reshape(-1,N)
            t, N = data.shape

        data_dim0 = data.shape[0]
        data_dim1 = data.shape[1]
        if data_dim0 < window_size:
            data_dim_pad_num = window_size - data_dim0
            data_dim1_pad = np.zeros([data_dim_pad_num,data_dim1])
            data = np.concatenate((data, data_dim1_pad), axis=0)

        data = data.T
        v = np.lib.stride_tricks.sliding_window_view(data, (N, window_size))
        v = v.squeeze(axis =0)
        #mm = np.max(v, axis=2)

        if t > window_size:
            label=np.zeros([t-window_size+1,N])
        else:
            label = np.zeros([window_size + 1 - t , N])

        for i in range(len(v)):
            for j in range(len(v[i])):
                aa=v[i][j]
                bb=np.max(v[i][j])
                zhangfu = (np.max(v[i][j]) - v[i][j][0]) / v[i][j][0]
                if zhangfu>0.00001:
                    label[i][j] = 1
                else:
                    label[i][j] = 0
        label = torch.tensor(label, dtype=torch.float)
        return label

def label_new(data,window_size,index):
    if torch.is_tensor(data):
        b,t,N=data.size()
        data=data.view(1,-1,N).squeeze()
        inc=torch.zeros(data.shape)
        for i in range(1,len(data)):
            inc[i,:]=data[i,:]-data[i-1,:]
            label=torch.where(inc > 0, 1, 0)
        label=label.view(b, t, N )
        label=torch.tensor(label,dtype=torch.float)
        return label
    elif type(data) == np.ndarray:
        if data.ndim==2:
            t, N = data.shape
        elif data.ndim==3:
            b,t,N = data.shape
            data=data.reshape(-1,N)
            t, N = data.shape
        data = data.T
        if index is not None and index !=[]:
            index = [i[:-1] for i in index]
            label = np.zeros([len(index[0]), N])
            for i in range(len(data)):
                index_i = index[i]
                label = np.zeros([len(index_i), N])
                for j in range(len(index_i)-1):
                    if index_i[j]+window_size >=len(data[0]):
                        aa=list(data[i,index_i[j]:])
                    else:
                        aa=list(data[i,index_i[j]:index_i[j]+window_size])
                    zhangfu=(aa[-1]-aa[0])/aa[0]
                    if zhangfu>0.00001:
                        label[j][i] = 1
            label=label.T
        else:
            v = np.lib.stride_tricks.sliding_window_view(data, (N, window_size))
            v = v.squeeze()
            label=np.zeros([len(v),len(v[0])])  #t-window_size+1,N
            for i in range(len(v)):
                for j in range(len(v[0])):
                    zhangfu=(np.max(v[i][j])-v[i][j][0])/v[i][j][0]
                    if zhangfu>0.00001:
                        label[i][j] = 1
                    else:
                        label[i][j] = 0
        label = torch.tensor(label, dtype=torch.float)
        return label
#
def inference(model, dataloader, device, node_cnt, window_size, horizon):
    forecast_set = []
    target_set = []
    model.eval()

    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            step = 0
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)

            while step < horizon:
                forecast_result, a = model(inputs)
                len_model_output = forecast_result.size()[1]
                if len_model_output == 0:
                    raise Exception('Get blank inference result')
                inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                                   :].clone()
                inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
                step += min(horizon - step, len_model_output)

            forecast_set.append(forecast_steps)
            target_set.append(target.detach().cpu().numpy())
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)

def validate(model, dataloader, device, normalize_method, statistic, node_cnt, window_size, horizon,Regression,
             Classification, zdfl_windows, result_file=None):
    forecast_norm, target_norm = inference(model, dataloader, device, node_cnt, window_size, horizon)
    if Regression:
        start = datetime.now()

        if normalize_method and statistic:
            forecast = de_normalized(forecast_norm, normalize_method, statistic)
            target = de_normalized(target_norm, normalize_method, statistic)
        else:
            forecast, target = forecast_norm, target_norm

        b, t, N = forecast.shape
        forecast_de = forecast.reshape(-1, N)
        target_de = target.reshape( -1, N)

        gap_forecast = compute_ext(torch.tensor(forecast, dtype=torch.float), 20, mode='不重叠窗口')#.reshape(b,-1,N)
        gap_target = compute_ext(torch.tensor(target, dtype=torch.float), 20, mode='不重叠窗口')#.reshape(b,-1,N)
        ma_forecast = MA(torch.tensor(forecast, dtype=torch.float), 20)#.reshape(b,-1,N)
        ma_target = MA(torch.tensor(target, dtype=torch.float), 20)#.reshape(b,-1,N)

        results1 = []
        results2 = []
        for j in range(len(forecast_de[0])):
            tt_algo = rpt.Dynp(model="l2").fit(target_de[:, j])
            index_target = tt_algo.predict(10)
            tt_algo2 = rpt.Dynp(model="l2").fit(forecast_de[:, j])
            index_predict=tt_algo2.predict(10)
            results1.append(index_target)
            results2.append(index_predict)
        #
        score = evaluate(Regression,False,target, forecast)
        score_by_node = evaluate(Regression,False,target, forecast, by_node=True)
        end = datetime.now()
        score_norm = evaluate(Regression,False,target_norm, forecast_norm)
        print(f'NORM: MAPE {score_norm[0]:7.9}; MAE {score_norm[1]:7.9f}; RMSE {score_norm[2]:7.9f}.')
        print(f'RAW : MAPE {score[0]:7.9}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
        if result_file:
            if not os.path.exists(result_file):
                os.makedirs(result_file)

            step_to_print = 0
            forcasting_2d = forecast.reshape(1, -1, N).squeeze()
            forcasting_2d_target = target.reshape(1, -1, N).squeeze() #[:, step_to_print, :]

            np.savetxt(f'{result_file}/target.csv', forcasting_2d_target, delimiter=",")
            np.savetxt(f'{result_file}/predict.csv', forcasting_2d, delimiter=",")
            np.savetxt(f'{result_file}/predict_abs_error.csv',
                       np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
            np.savetxt(f'{result_file}/predict_ape.csv',
                       np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")

            # np.savetxt(f'{result_file}/gap_target.csv',gap_target, delimiter=",")
            # np.savetxt(f'{result_file}/gap_forecast.csv', gap_forecast, delimiter=",")
            # np.savetxt(f'{result_file}/ma_forecast.csv',ma_forecast, delimiter=",")
            # np.savetxt(f'{result_file}/ma_target2d.csv',ma_target , delimiter=",")
            # np.savetxt(f'{result_file}/changepoint_target.csv',results1, delimiter=",")
            # np.savetxt(f'{result_file}/changepoint_predict.csv', results2, delimiter=",")

        # ff = forecast_de[:, ].tolist()
        # tt = target_de[:, ].tolist()
        #
        # for j in range(len(ff[0])):
        #     tt_de = [i[j] for i in tt]
        #     ff_de = [i[j] for i in ff]
        #     line1, = plt.plot(tt_de, color='b')
        #     line2, = plt.plot(ff_de, color='r')
        #     plt.legend(handles=[line1, line2], labels=['target', 'forecast'], loc='upper right')
        #     plt.title(str(j))
        #     plt.xlabel('tick')
        #     plt.savefig('./png/forecast' + str(j) + '.png', format='png', dpi=300)
        #     plt.show()
        #
        # for j in range(len(gap_forecast[0])):
        #     tt_de = [i[j] for i in gap_target]
        #     ff_de = [i[j] for i in gap_forecast]
        #     line1, = plt.plot(tt_de, color='b')
        #     line2, = plt.plot(ff_de, color='r')
        #     plt.legend(handles=[line1, line2], labels=['target', 'forecast'], loc='upper right')
        #     plt.title(str(j))
        #     plt.xlabel('tick')
        #     plt.savefig('./png/gap' + str(j) + '.png', format='png', dpi=300)
        #     plt.show()
        # for j in range(len(ma_forecast[0])):
        #     tt_de = [i[j] for i in ma_target]
        #     ff_de = [i[j] for i in ma_forecast]
        #     line1, = plt.plot(tt_de, color='b')
        #     line2, = plt.plot(ff_de, color='r')
        #     plt.legend(handles=[line1, line2], labels=['target', 'forecast'], loc='upper right')
        #     plt.title(str(j))
        #     plt.xlabel('tick')
        #     plt.savefig('./png/MA' + str(j) + '.png', format='png', dpi=300)
        #     plt.show()

        return dict(mae=score[1], mae_node=score_by_node[1], mape=score[0], mape_node=score_by_node[0],
                    rmse=score[2], rmse_node=score_by_node[2])
    if Classification:
        start = datetime.now()
        if normalize_method and statistic:
            forecast = de_normalized(forecast_norm, normalize_method, statistic)
            target = de_normalized(target_norm, normalize_method, statistic)
            forecast_label = label(forecast,zdfl_windows)
            target_label = label(target, zdfl_windows)
            forecast_label=forecast_label.detach().cpu().numpy().ravel(order='F')
            target_label=target_label.detach().cpu().numpy().ravel(order='F')

            fpr,tpr,thre=roc_curve(forecast_label,target_label,pos_label=1)
            auc1= auc(fpr,tpr)
            # print(fpr,tpr,thre)
            # print(auc(fpr,tpr))

            # plt.plot(fpr, tpr, color='darkred', label='roc area:(%0.2f)' % auc1)
            # plt.plot([0, 1], [0, 1], linestyle='--')
            # plt.xlim([0, 1])
            # plt.ylim([0, 1])
            # plt.xlabel('fpr')
            # plt.ylabel('tpr')
            # plt.title('roc_curve')
            # plt.legend(loc='lower right')
            # plt.show()
        else:
            forecast, target = forecast_norm, target_norm

        score = evaluate(False,Classification, target_label, forecast_label)
        if True in np.isnan(score):
            score = np.nan_to_num(score, nan=0.5)

        end = datetime.now()
        score_norm = evaluate(False,Classification, target_label, forecast_label)
        # print(f'NORM: precision {score_norm[0]:7.9}; recall {score_norm[1]:7.9f};accuarcy {score_norm[2]:7.9f};f1 {score_norm[3]:7.9f}.')
        print(f'RAW : precision {score[0]:7.9}; recall {score[1]:7.9f}; accuarcy {score[2]:7.9f};f1 {score[3]:7.9f}.')
        if result_file:
            if not os.path.exists(result_file):
                os.makedirs(result_file)
            step_to_print = 0
            forcasting_2d = forecast[:, step_to_print, :]
            forcasting_2d_target = target[:, step_to_print, :]

            np.savetxt(f'{result_file}/target2.csv', forcasting_2d_target, delimiter=",")
            np.savetxt(f'{result_file}/predict2.csv', forcasting_2d, delimiter=",")
            np.savetxt(f'{result_file}/predict_abs_error.csv',
                       np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
            np.savetxt(f'{result_file}/predict_ape.csv',
                       np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")
        return dict(recall=score[1], precision=score[0],
                    accuarcy=score[2], f1=score[3])

def infoNCE(in_features1, in_features2, batchsize, temperature=0.9, eps=1e-8):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    features_t = torch.transpose(in_features2, dim0=0, dim1=1)
    features_c= torch.mm(in_features1, features_t)
    g_matrix = torch.exp(features_c/temperature)
    g_matrix1=torch.where(torch.isinf(g_matrix), torch.full_like(g_matrix, 0), g_matrix)

    denom = torch.sum(g_matrix1, dim=0, keepdim=True)
    sim_matrix = -torch.log(g_matrix / denom )
    sim_matrix1 = torch.where(torch.isinf(sim_matrix), torch.full_like(sim_matrix, 0), sim_matrix)
    logB = torch.log(torch.Tensor([batchsize]))

    MILoss = (logB + torch.sum(sim_matrix1)/(batchsize*batchsize)).to(device)

    return MILoss

def infoNCE2D(in_features1, in_features2, batchsize, temperature=0.9, eps=1e-8):
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    features_t = torch.transpose(in_features2, dim0=0, dim1=1)
    features_c= torch.mm(features_t, in_features1)

    logits_max, _ = torch.max(features_c, dim=1, keepdim=True)
    sim_matrix11 = features_c - logits_max.detach()
    B = features_c.size(0)
    eye = torch.eye(B)  # (B', B')

    g_matrix = torch.exp(sim_matrix11/temperature)*(1 - eye)
    g_matrix1=torch.where(torch.isinf(g_matrix), torch.full_like(g_matrix, 0), g_matrix)

    denom = torch.sum(g_matrix1, dim=0, keepdim=True)
    sim_matrix = -torch.log(g_matrix / (denom + eps) + eps)

    sim_matrix1 = torch.where(torch.isinf(sim_matrix), torch.full_like(sim_matrix, 0), sim_matrix)

    a_diag = torch.diag_embed(torch.diagonal(sim_matrix1, 0))
    diag1 = (sim_matrix1 - a_diag)

    logB = torch.log(torch.Tensor([batchsize]))
    MILoss = (logB + torch.sum(diag1) / (B*B)).to(device)

    return MILoss

def infoNCE3D(in_features1, in_features2, batchsize, temperature=0.99, eps=1e-8):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    features_t = torch.transpose(in_features1, dim0=1, dim1=2)
    features_c= torch.bmm(in_features2,features_t)  #AT*A

    g_matrix = torch.exp(features_c /temperature)
    denom = torch.sum(g_matrix, dim=1, keepdim=True)

    sim_matrix = -torch.log(g_matrix / (denom + eps) + eps)  # loss matrix
    logB = torch.log(torch.Tensor([batchsize]))
    MILoss = (logB + torch.sum(sim_matrix) / (batchsize*batchsize)).to(device)

    return MILoss

def infoNCE3D2(in_features1, in_features2,batchsize, temperature=0.99, eps=1e-8):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    features_t = torch.transpose(in_features1, dim0=1, dim1=2)
    features_c= torch.bmm(features_t, in_features2)  #AT*A
    #print('----features_c--------',features_c.shape)

    g_matrix = torch.exp(features_c /temperature)
    denom = torch.sum(g_matrix, dim=1, keepdim=True)

    sim_matrix = -torch.log(g_matrix / (denom + eps) + eps)  # loss matrix
    logB = torch.log(torch.Tensor([batchsize]))
    MILoss =(logB + torch.sum(sim_matrix) / (batchsize*batchsize) ).to(device)

    return MILoss

def gnn_model_validate(model,valid_loader,node_cnt, epoch, args,best_validate_mae,validate_score_non_decrease_count,
                        best_validate_f1,result_file, normalize_statistic):

    if (epoch + 1) % args.validate_freq == 0:
        is_best_for_now = False
        print('------ validate on data: VALIDATE ------')
        performance_metrics1 = \
            validate(model, valid_loader, args.device, args.norm_method, normalize_statistic,
                     node_cnt, args.window_size, args.horizon, args.Regression, False, args.zdfl_windows,
                     result_file=result_file)
        performance_metrics2 = validate(model, valid_loader, args.device, args.norm_method, normalize_statistic,
                                        node_cnt, args.window_size, args.horizon, False, args.Classification,
                                        args.zdfl_windows, result_file=None)
        if args.Regression:
            if best_validate_mae > performance_metrics1['mae']:
                best_validate_mae = performance_metrics1['mae']
                is_best_for_now = True
                validate_score_non_decrease_count = 0
            else:
                validate_score_non_decrease_count += 1
        if args.Classification:
            if best_validate_f1 < performance_metrics2['f1']:
                best_validate_f1 = performance_metrics2['f1']
                is_best_for_now = True
                validate_score_non_decrease_count = 0
            else:
                validate_score_non_decrease_count += 1
        # save model
        if is_best_for_now:
            save_model(model, result_file)

        return validate_score_non_decrease_count

def gnn_model_optimizer_main(train_data, result_file,args):
    if args.norm_method == 'z_score':
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == 'min_max':
        train_min = np.min(train_data, axis=0)
        train_max = np.max(train_data, axis=0)
        normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        normalize_statistic = None
    if normalize_statistic is not None:
        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
            json.dump(normalize_statistic, f)

    return normalize_statistic

def gnn_optimizer(model, args):
    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08, weight_decay=0.001)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.001)
    return my_optim

def data_train_valid_loader(train_data, valid_data, args, normalize_statistic):
    train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=0)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader,valid_loader

def train_data_pad(forecast,forecast_aux, target, args):
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    batchsize_aux = forecast_aux.shape[0]
    c = forecast.shape[1]
    d = forecast.shape[2]
    t2 = 0
    if batchsize_aux < args.batch_size:
        num = args.batch_size - batchsize_aux
        t2 = torch.ones(num, c, d).to(device)
        forecast_aux=forecast_aux.to(device)
        forecast_aux = torch.cat((forecast_aux, t2), dim=0)

    batchsize_main = forecast.shape[0]
    forecast_c = forecast.shape[1]
    forecast_d = forecast.shape[2]
    t_main = 0
    if batchsize_main < args.batch_size:
        num_main = args.batch_size - batchsize_main
        t_main = torch.ones(num_main, forecast_c, forecast_d) .to(device)
        forecast = torch.cat((forecast, t_main), dim=0)

    target_main = target.shape[0]
    target_c = target.shape[1]
    target_d = target.shape[2]
    target_t = 0
    if target_main < args.batch_size:
        num_target = args.batch_size - target_main
        target_t = torch.ones(num_target, target_c, target_d).to(device)
        target = torch.cat((target, target_t), dim=0)

    return forecast,forecast_aux, target

def  train_ma_task(model, forecast, forecast_aux, batch_size, loss_total_ma, my_optim, args):

    forecast_ma= MA (forecast,args.ma_windows) #forecast_ma shape: [141, 64]
    forecast_ma_aux = MA(forecast_aux, args.ma_windows)  # forecast_ma_aux shape: [141, 64]

    # Mutual information
    LossMoveA_MI = infoNCE2D(forecast_ma, forecast_ma_aux, batch_size, temperature=0.99, eps=1e-8)
    LossMoveA_MI.requires_grad_(True)
    LossMoveA_MI.backward(retain_graph=True)

    regularization_loss = 0
    for para in model.parameters():
        #if p.grad is not None:
        regularization_loss += torch.norm(para)

    penalty = args.penalty_para * regularization_loss
    loss_sum_ma = LossMoveA_MI  + penalty

    loss_sum_ma.backward()
    my_optim.step()
    loss_total_ma += float(loss_sum_ma)

    return loss_sum_ma

def train_ma_task_current_para(model,loss_sum_ma, b_ma, my_lr_scheduler,epoch, args, result_file):
    if (epoch + 1) % args.exponential_decay_step == 0:
        my_lr_scheduler.step()

    b_ma.append(loss_sum_ma.item())

    theta_loss_sum_ma = 0
    for i, p in enumerate(model.parameters()):  # i = 73
        omega_theta = 0.04
        omega_theta = omega_theta * (p - model.omega_theta_optimal_para_MA[0][i]).pow(2)
        theta_loss_sum_ma += omega_theta.sum()

    model.omega_w_current_para_MA[model.MA_current_task] = []
    w_loss_sum = 0
    for name, parameter in model.named_parameters():
        if "weight" in name:
            pw = parameter.data.clone()
            model.omega_w_current_para_MA[model.MA_current_task].append(pw)

    w_current_ma = [a - b for a, b in zip(model.omega_w_current_para_MA[0], model.omega_w_optimal_para_MA[0])]
    w_current_pow_ma = [num.pow(2) for num in w_current_ma]
    w_total_ma = 0
    for inputs_index, target_value in enumerate(w_current_pow_ma):
        w_tensor = target_value.sum()
        w_total_ma += w_tensor.sum()

    return theta_loss_sum_ma, w_total_ma


def train_gap_task(model,forecast,forecast_aux, batch_size,loss_total_gap, my_optim, args):

    # 计算极值
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    forecast_ext=compute_ext(forecast, args.jicha_windows, mode='不重叠窗口')#forecast_ext shape: [8, 64]
    forecast_ext_aux = compute_ext(forecast_aux, args.jicha_windows, mode='不重叠窗口')  # forecast_ext_aux shape: [8, 64]

    # Mutual information
    Loss_gap_MI = infoNCE(forecast_ext, forecast_ext_aux, batch_size, temperature=0.99, eps=1e-8).to(device)
    Loss_gap_MI.requires_grad_(True)
    Loss_gap_MI.backward(retain_graph=True)

    regularization_loss = 0
    for para in model.parameters():
        #if p.grad is not None:
        regularization_loss += torch.norm(para)

    penalty = args.penalty_para * regularization_loss
    loss_sum_gap = Loss_gap_MI  + penalty

    loss_sum_gap.backward()
    my_optim.step()
    loss_total_gap += float(loss_sum_gap)

    return loss_sum_gap

def train_gap_task_current_para(model,loss_sum_gap, b_gap, my_lr_scheduler,epoch, args, result_file):
    if (epoch + 1) % args.exponential_decay_step == 0:
        my_lr_scheduler.step()

    b_gap.append(loss_sum_gap.item())

    theta_loss_sum_gap = 0
    for i, p in enumerate(model.parameters()):  # i = 73
        omega_theta = 0.04
        omega_theta = omega_theta * (p - model.omega_theta_optimal_para_gap[0][i]).pow(2)
        theta_loss_sum_gap += omega_theta.sum()

    model.omega_w_current_para_gap[model.gap_current_task] = []
    w_loss_sum = 0
    for name, parameter in model.named_parameters():
        if "weight" in name:
            pw = parameter.data.clone()
            model.omega_w_current_para_gap[model.gap_current_task].append(pw)
    w_current_ma = [a - b for a, b in zip(model.omega_w_current_para_gap[0], model.omega_w_optimal_para_gap[0])]
    w_current_pow_ma = [num.pow(2) for num in w_current_ma]
    w_total_gap = 0
    for inputs_index, target_value in enumerate(w_current_pow_ma):
        w_tensor = target_value.sum()
        w_total_gap += w_tensor.sum()

    return theta_loss_sum_gap, w_total_gap

def train_cpd_task(model, forecast, forecast_aux, batch_size, loss_total_cpd, N, my_optim, args, normalize_statistic):
    # forecast_de.shape: [160, 64]
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    forecast_de = de_normalized(forecast.view(1, -1, N).squeeze().detach().cpu().numpy(), args.norm_method, normalize_statistic)
    #forecast_de_aux.shape: [160, 64]
    forecast_de_aux = de_normalized(forecast_aux.view(1, -1, N).squeeze().detach().cpu().numpy(), args.norm_method, normalize_statistic)

    # Mutual information
    forecast_label = label(forecast_de, args.zdfl_windows)  # forecast_label shape: [81, 64]
    forecast_label_aux = label(forecast_de_aux, args.zdfl_windows)  # forecast_label_aux shape: [81, 64]
    loss4_MI = infoNCE2D(forecast_label, forecast_label_aux, batch_size, temperature=0.99, eps=1e-8).to(device)
    loss4_MI.requires_grad_(True)
    loss4_MI.backward(retain_graph=True)

    regularization_loss = 0
    for para in model.parameters():
        #if p.grad is not None:
        #regularization_loss += torch.sum(torch.abs(p))
        regularization_loss += torch.norm(para)
    penalty = args.penalty_para * regularization_loss

    loss_sum_cpd = loss4_MI  + penalty

    loss_sum_cpd.backward()
    my_optim.step()
    loss_total_cpd += float(loss_sum_cpd)
    return loss_sum_cpd

def train_cpd_task_current_para(model,loss_sum_cpd, b_cpd, my_lr_scheduler,epoch, args, result_file):
    if (epoch + 1) % args.exponential_decay_step == 0:
        my_lr_scheduler.step()

    b_cpd.append(loss_sum_cpd.item())

    theta_loss_sum = 0
    for i, p in enumerate(model.parameters()):  # i = 73
        omega_theta = 0.04
        omega_theta = omega_theta * (p - model.omega_theta_optimal_para_cpd[0][i]).pow(2)
        theta_loss_sum += omega_theta.sum()

    model.omega_w_current_para_cpd[model.cpd_current_task] = []
    w_loss_sum = 0
    for name, parameter in model.named_parameters():
        if "weight" in name:
            pw = parameter.data.clone()
            model.omega_w_current_para_cpd[model.cpd_current_task].append(pw)

    w_current_ma = [a - b for a, b in zip(model.omega_w_current_para_cpd[0], model.omega_w_optimal_para_cpd[0])]
    w_current_pow_ma = [num.pow(2) for num in w_current_ma]
    w_total = 0
    for inputs_index, target_value in enumerate(w_current_pow_ma):
        w_tensor = target_value.sum()
        w_total += w_tensor.sum()

    return theta_loss_sum, w_total


def get_theta_w_paramete(model, tasklist, taskname):

    if tasklist == 'task1' and taskname == 'cpd':
        model.omega_theta_optimal_para_cpd[model.cpd_current_task] = []
        for parameter in model.parameters():
            pd = parameter.data.clone()
            model.omega_theta_optimal_para_cpd[model.cpd_current_task].append(pd)

        model.omega_w_optimal_para_cpd[model.cpd_current_task] = []
        for name, parameter in model.named_parameters():
            if "weight" in name:
                # print(name, parameter.grad.data.clone())
                pw = parameter.data.clone()
                model.omega_w_optimal_para_cpd[model.cpd_current_task].append(pw)
    if tasklist == 'task2' and taskname == 'gap':
        model.omega_theta_optimal_para_gap[model.gap_current_task] = []
        for parameter in model.parameters():
            pd = parameter.data.clone()
            model.omega_theta_optimal_para_gap[model.gap_current_task].append(pd)

        model.omega_w_optimal_para_gap[model.gap_current_task] = []
        for name, parameter in model.named_parameters():
            if "weight" in name:
                pw = parameter.data.clone()
                model.omega_w_optimal_para_gap[model.gap_current_task].append(pw)

    if tasklist == 'task3' and taskname == 'MA':
        model.omega_theta_optimal_para_MA[model.MA_current_task] = []
        for parameter in model.parameters():
            pd = parameter.data.clone()
            model.omega_theta_optimal_para_MA[model.MA_current_task].append(pd)

        model.omega_w_optimal_para_MA[model.MA_current_task] = []
        for name, parameter in model.named_parameters():
            if "weight" in name:
                pw = parameter.data.clone()
                model.omega_w_optimal_para_MA[model.MA_current_task].append(pw)

    if tasklist == 'task4' and taskname == 'forecast':
        return

def train_forecast_task(model, forecast, target, loss_total, forecast_loss, my_optim,args):
        Loss_forecast = forecast_loss(forecast, target)

        regularization_loss = 0
        for p in model.parameters():
            #if p.grad is not None:
            #regularization_loss += torch.sum(torch.abs(p))
            regularization_loss += torch.norm(p)

        penalty = args.penalty_para * regularization_loss
        loss_sum = Loss_forecast + penalty

        loss_sum.backward()
        my_optim.step()
        loss_total += float(loss_sum)

        return loss_sum,Loss_forecast,loss_total

def  train_forecast_task_validate(model,node_cnt, valid_loader, my_lr_scheduler, loss_total, cnt,epoch, args, result_file,epoch_start_time,
                            best_validate_mae, validate_score_non_decrease_count, best_validate_f1,normalize_statistic,a_forecast,
                                              b_forecast, Loss_forecast,loss_sum_forecast):

    print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch,
                                             (time.time() - epoch_start_time), loss_total / cnt))

    save_model(model, result_file, epoch)
    if (epoch + 1) % args.exponential_decay_step == 0:
        my_lr_scheduler.step()

    validate_score_non_decrease_count = gnn_model_validate(model, valid_loader, node_cnt, epoch, args,
                                                           best_validate_mae,
                                                           validate_score_non_decrease_count, best_validate_f1,
                                                           result_file,
                                                           normalize_statistic)
    # early stop
    if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
        return

    a_forecast.append(Loss_forecast.item())
    b_forecast.append(loss_sum_forecast.item())

def data_train_valid_loader_aux(train_data, valid_data, args, normalize_statistic):
    train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=0)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader,valid_loader

def gnn_model_optimizer_aux(train_data_aux, result_file_aux,args):
    if args.norm_method == 'z_score':
        train_mean = np.mean(train_data_aux, axis=0)
        train_std = np.std(train_data_aux, axis=0)
        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == 'min_max':
        train_min = np.min(train_data_aux, axis=0)
        train_max = np.max(train_data_aux, axis=0)
        normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        normalize_statistic = None
    if normalize_statistic is not None:
        with open(os.path.join(result_file_aux, 'norm_stat_aux.json'), 'w') as f:
            json.dump(normalize_statistic, f)

    return normalize_statistic

def train_aux(train_data_aux, valid_data_aux, args, result_file_aux):
    node_cnt_aux = train_data_aux.shape[1]
    model_aux = Model(node_cnt_aux, 2, args.window_size, args.multi_layer, args.lambda_l, args.lambda_t, horizon=args.horizon)

    model_aux.to(args.device)
    if len(train_data_aux) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data_aux) == 0:
        raise Exception('Cannot organize enough validation data')

    normalize_statistic = gnn_model_optimizer_aux(train_data_aux, result_file_aux, args)

    my_optim_aux = gnn_optimizer(model_aux, args)
    my_lr_scheduler_aux = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim_aux, gamma=args.decay_rate)

    train_loader_aux, _ = data_train_valid_loader_aux(train_data_aux, valid_data_aux, args, normalize_statistic)

    if args.Regression :
        forecast_loss = nn.MSELoss(reduction='mean').to(args.device)

    if args.Classification :
        forecast_loss2 = nn.BCEWithLogitsLoss(reduction='mean').to(args.device)
        #best_validate_f1 = 0

    a=[]
    b=[]
    loss_i= []
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model_aux.train()
        loss_total = 0
        cnt = 0
        loss_i.append([0]*76)

        for i, (inputs, target) in enumerate(train_loader_aux):  # i=17
            # inputs shape: [32, 120, 64]
            inputs = inputs.to(args.device)   # torch.size(32,12,76)  [batch_size,window_size,N] = [32,120,64]
            # target shape: [32, 120, 128]
            target = target.to(args.device)
            model_aux.zero_grad()

            forecast, _ = model_aux(inputs)
            _, _, N = forecast.size()   #forecast shape: [32, 5, 64]

            forecast_de = de_normalized(forecast.view(1, -1, N).squeeze().detach().cpu().numpy(), args.norm_method, normalize_statistic)
            target_de = de_normalized(target.view(1, -1,N).squeeze().detach().cpu().numpy(), args.norm_method, normalize_statistic)

            forecast_ma= MA (forecast,args.ma_windows)#forecast_ma shape: [141, 64]
            target_ma = MA (target,args.ma_windows)   #target_ma shape: [141, 64]

            forecast_ext=compute_ext(forecast, args.jicha_windows, mode='不重叠窗口')#forecast_ext shape: [8, 64]
            target_ext=compute_ext(target, args.jicha_windows, mode='不重叠窗口') #target_ext shape: [8, 64]

            Loss_lag = forecast_loss(forecast_ext, target_ext) #forecast_ext shape: [8, 64]
            Loss_lag .requires_grad_(True)

            LossMoveA= forecast_loss(forecast_ma, target_ma) #forecast_ma shape: [141, 64]
            LossMoveA.requires_grad_(True)

            forecast_label=label(forecast_de,args.zdfl_windows)  #forecast_label shape: [81, 64]
            target_label=label(target_de,args.zdfl_windows)      #target_label shape: [81, 64]
            loss4 = forecast_loss2(forecast_label, target_label)
            loss4.requires_grad_(True)

            Loss_forecast = forecast_loss(forecast, target)
            Loss_forecast.requires_grad_(True)

            cnt += 1
            loss_sum =   LossMoveA+Loss_lag+loss4+Loss_forecast # loss_sum + theta_sum + w_sum


            loss_sum.backward()
            my_optim_aux.step()
            loss_total += float(loss_sum)

        save_model(model_aux, result_file_aux, epoch)
        if (epoch+1) % args.exponential_decay_step == 0:
            my_lr_scheduler_aux.step()

        a.append(Loss_forecast.item())
        b.append(loss_sum.item())

    return forecast

def plt_cpd_display(tasklist, taskname, t_forecast, f_forecast):
    if tasklist == 'task4' and taskname == 'forecast':
        for j in range(len(t_forecast[0])):
            tt_de = [i[j] for i in t_forecast]
            ff_de = [i[j] for i in f_forecast]
            line1, = plt.plot(tt_de, color='b')
            line2, = plt.plot(ff_de, color='r')
            plt.legend(handles=[line1, line2], labels=['target', 'forecast'], loc='upper right')
            plt.title('trend' + str(j))
            plt.xlabel('tick')
            plt.savefig('./png/forecast' + str(j) + '.png', format='png', dpi=300)
            plt.show()


def task_train2(train_data,train_data_aux, valid_data,valid_data_aux, args, result_file):
    node_cnt = train_data.shape[1]
    model = Model(node_cnt, 2, args.window_size, args.multi_layer, args.lambda_l, args.lambda_t, horizon=args.horizon)

    model.to(args.device)
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')

    normalize_statistic = gnn_model_optimizer_main(train_data, result_file, args)

    my_optim = gnn_optimizer(model, args)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_loader, valid_loader = data_train_valid_loader(train_data, valid_data, args, normalize_statistic)

    if args.Regression :
        forecast_loss = nn.MSELoss(reduction='mean').to(args.device)
        best_validate_mae = np.inf

    if args.Classification :
        forecast_loss2 = nn.BCEWithLogitsLoss(reduction='mean').to(args.device)
        best_validate_f1 = 0

    total_params = 0
    validate_score_non_decrease_count = 0
    performance_metrics = {}
    a_forecast = []
    b_forecast = []
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        cnt = 0
        loss_sum = 0
        loss_sum_forecast = 0
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(args.device)  #torch.size(32,12,76)  [batch_size,window_size,N)
            target = target.to(args.device)
            model.zero_grad()
            forecast, _ = model(inputs)
            _, _, N = forecast.size()
            forecast_de = de_normalized(forecast.view(1, -1,N).squeeze().detach().cpu().numpy(), args.norm_method, normalize_statistic)
            target_de = de_normalized(target.view(1, -1,N).squeeze().detach().cpu().numpy(), args.norm_method, normalize_statistic)

            loss1 = forecast_loss(forecast, target)

            forecast_ma = MA(forecast, args.ma_windows)
            target_ma = MA(target, args.ma_windows)
            loss3 = forecast_loss(forecast_ma, target_ma)

            forecast_ext = compute_ext(forecast, args.jicha_windows, mode='不重叠窗口')
            target_ext = compute_ext(target, args.jicha_windows, mode='不重叠窗口')
            loss2 = forecast_loss(forecast_ext, target_ext)

            forecast_label = label(forecast_de, args.zdfl_windows)  # results
            target_label = label(target_de, args.zdfl_windows)  # pre_results
            loss4 = forecast_loss2(forecast_label, target_label)

            total_task_loss_sum=loss1+loss2+loss3+loss4
            loss1.backward(retain_graph=True)
            my_optim.step()

            loss_sum, loss_sum_forecast, loss_total = train_forecast_task(model, forecast, target, 0,
                                                                          forecast_loss, my_optim, args)
            train_forecast_task_validate(model, node_cnt, valid_loader, my_lr_scheduler, total_task_loss_sum, cnt,
                                         epoch, args,
                                         result_file, epoch_start_time, best_validate_mae,
                                         validate_score_non_decrease_count,
                                         best_validate_f1, normalize_statistic, a_forecast, b_forecast, loss_sum,
                                         loss_sum_forecast)
    return performance_metrics, normalize_statistic

def task_train(train_data,train_data_aux, valid_data,valid_data_aux, args, result_file,result_file_aux):
    forecast_aux = train_aux(train_data_aux, valid_data_aux, args, result_file_aux)  #forecast_aux.shape: [32, 5, 64]
    #train_data.shape:  (700, 64)
    node_cnt = train_data.shape[1]
    model = Model(node_cnt, 2, args.window_size, args.multi_layer, args.lambda_l, args.lambda_t, horizon=args.horizon)

    model.to(args.device)
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')

    normalize_statistic = gnn_model_optimizer_main(train_data, result_file, args)

    my_optim = gnn_optimizer(model, args)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_loader, valid_loader = data_train_valid_loader(train_data, valid_data, args, normalize_statistic)

    if args.Regression :
        forecast_loss = nn.MSELoss(reduction='mean').to(args.device)
        best_validate_mae = np.inf

    if args.Classification :
        forecast_loss2 = nn.BCEWithLogitsLoss(reduction='mean').to(args.device)
        best_validate_f1 = 0

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")
    validate_score_non_decrease_count = 0
    performance_metrics = {}

    for tasklist, taskname in model.multitasklist.items():
        get_theta_w_paramete(model, tasklist, taskname)

        a_forecast= []  #forecast
        b_forecast = [] #forecast

        b_cpd= []  #cpd
        b_gap= []  #gap
        b_ma = []  #MA
        #forecast_f = []  # cpd:forecast
        #target_f = []  # cpd:target

        #model.forecast_f = []
        #model.target_f = []

        loss_i= []
        for epoch in range(args.epoch):
            epoch_start_time = time.time()
            model.train()

            loss_total = 0
            loss_total_cpd = 0
            loss_total_gap = 0
            loss_total_ma = 0
            loss_total_forecast = 0
            loss_sum = 0
            loss_sum_forecast = 0

            theta_loss_sum_gap = 0
            theta_loss_sum_ma = 0
            theta_loss_sum_cpd = 0
            w_total_gap = 0
            w_total_ma = 0
            w_total_cpd = 0

            cnt = 0
            loss_i.append([0]*node_cnt)

            for i, (inputs, target) in enumerate(train_loader):  # i=17
                # inputs shape: [32, 120, 64]
                inputs = inputs.to(args.device)   # torch.size(32,12,76)  [batch_size,window_size,N] = [32,120,64]
                # target shape: [32, 120, 64]
                target = target.to(args.device)

                model.zero_grad()
                forecast, _ = model(inputs)
                _, _, N = forecast.size()   #forecast shape: [32, 5, 64]

                batch_size = args.batch_size
                cnt += 1

                forecast,forecast_aux, target = train_data_pad(forecast, forecast_aux, target, args)

                if tasklist == 'task1' and taskname == 'cpd':
                    loss_total_cpd = train_cpd_task(model, forecast, forecast_aux, batch_size, loss_total_cpd, N, my_optim, args, normalize_statistic)
                    break
                #
                if tasklist == 'task2' and taskname == 'gap':
                   loss_total_gap = train_gap_task(model,forecast,forecast_aux, batch_size,loss_total_gap, my_optim, args)
                   break
                if tasklist == 'task3' and taskname == 'MA':
                    loss_total_ma = train_ma_task(model, forecast, forecast_aux, batch_size, loss_total_ma, my_optim, args)
                    break
                if tasklist == 'task4' and taskname == 'forecast':
                    forecast_de = de_normalized(forecast.view(1, -1, N).squeeze().detach().cpu().numpy(), args.norm_method,  normalize_statistic)  # forecast_de.shape: [160, 64]
                    target_de = de_normalized(target.view(1, -1, N).squeeze().detach().cpu().numpy(), args.norm_method, normalize_statistic)  # target_de.shape: [160, 64]

                    model.forecast_f = forecast_de[:, ].tolist()
                    model.target_f = target_de[:, ].tolist()

                    loss_sum, loss_sum_forecast, loss_total = train_forecast_task(model, forecast, target,
                                                                                  loss_total_forecast, forecast_loss,
                                                                                  my_optim, args)

            if tasklist == 'task1' and taskname == 'cpd':
                theta_loss_sum_cpd, w_total_cpd = train_cpd_task_current_para(model, loss_total_cpd, b_cpd, my_lr_scheduler, epoch, args, result_file)
                print("---------loss_sum_cpd,theta_loss_sum, w_total------", loss_total_cpd, theta_loss_sum_cpd, w_total_cpd)
                break
            if tasklist == 'task2' and taskname == 'gap':
                theta_loss_sum_gap, w_total_gap = train_gap_task_current_para(model,loss_total_gap, b_gap, my_lr_scheduler,epoch, args, result_file)
                print("---------loss_sum_gap, theta_loss_sum_gap, w_total_gap-----", loss_total_gap, theta_loss_sum_gap,  w_total_gap)
                break
            if tasklist == 'task3' and taskname == 'MA':
                theta_loss_sum_ma, w_total_ma = train_ma_task_current_para(model, loss_total_ma, b_ma, my_lr_scheduler, epoch, args, result_file)
                print("---------loss_sum_ma, theta_loss_sum_ma, w_total_ma------", loss_total_ma, theta_loss_sum_ma, w_total_ma)
                break

            w_sum = args.lambda_l * (loss_total_gap * w_total_gap +loss_total_ma * w_total_ma) # loss_total_ma * w_total_ma + loss_total_gap * w_total_gap
            theta_sum = args.lambda_t * (loss_total_gap * w_total_gap +loss_total_ma * w_total_ma) #loss_total_ma * theta_loss_sum_ma ++ loss_total_gap * theta_loss_sum_gap

            # w_sum = args.lambda_l *(loss_total_ma * w_total_ma)# * (   loss_total_cpd * w_total_cpd) #loss_total_ma * w_total_ma +
            # theta_sum = args.lambda_t *(loss_total_ma * theta_loss_sum_ma)# * (  loss_total_cpd * theta_loss_sum_cpd)#loss_total_ma * theta_loss_sum_ma+

            total_task_loss_sum = loss_sum + w_sum+theta_sum

            print("------------total_task_loss_sum:    loss_total------------", total_task_loss_sum,loss_total)
            # total_task_loss_sum = loss_sum_forecast + loss_sum_cpd +  w_sum_cpd

            if tasklist == 'task4' and taskname == 'forecast':
                 Loss_forecast = forecast_loss(forecast, target)
                 train_forecast_task_validate(model, node_cnt, valid_loader, my_lr_scheduler, total_task_loss_sum, cnt, epoch, args,
                                                         result_file, epoch_start_time, best_validate_mae, validate_score_non_decrease_count,
                                                         best_validate_f1, normalize_statistic, a_forecast,b_forecast, loss_sum, loss_sum_forecast)

        #plt_cpd_display(tasklist, taskname, forecast, target)
        #if tasklist == 'task4' and taskname == 'forecast':
    '''
    for j in range(len(model.target_f[0])):
        tt_de = [i[j] for i in model.target_f]
        ff_de = [i[j] for i in model.forecast_f]
        line1, = plt.plot(tt_de, color='b')
        line2, = plt.plot(ff_de, color='r')
        plt.legend(handles=[line1, line2], labels=['target', 'forecast'], loc='upper right')
        plt.title('trend' + str(j))
        plt.xlabel('tick')
        plt.savefig('./png/forecast' + str(j) + '.png', format='png', dpi=300)
        plt.show()
    '''

    return performance_metrics, normalize_statistic

def test(test_data, args, result_train_file, result_test_file):
    #device = 'cuda:0'
    #device = torch.device(device if torch.cuda.is_available() else "cpu")

    with open(os.path.join(result_train_file, 'norm_stat.json'),'r') as f:
        normalize_statistic = json.load(f)

    model = load_model(result_train_file)
    node_cnt = test_data.shape[1]    # test_data.shape:  [101, 64]

    test_set = ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon, normalize_method='z_score', norm_statistic=normalize_statistic)
    test_loader = torch_data.DataLoader(test_set, batch_size=32, drop_last=False,shuffle=False, num_workers=0)

    performance_metrics1 = validate(model, test_loader, args.device, args.norm_method, normalize_statistic, node_cnt, args.window_size, args.horizon,args.Regression, False, args.zdfl_windows,result_file=result_test_file)
    performance_metrics2 = validate(model, test_loader, args.device, args.norm_method, normalize_statistic, node_cnt, args.window_size, args.horizon, False, args.Classification, args.zdfl_windows, result_file=None)

    if args.Regression:
        mae, mape, rmse = performance_metrics1['mae'], performance_metrics1['mape'], performance_metrics1['rmse']
        print('Performance on test set: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.4f}'.format(mape, mae, rmse))
    if args.Classification:
        precision, recall, accuarcy, f1 = performance_metrics2['precision'], performance_metrics2['recall'],\
                                       performance_metrics2['accuarcy'], performance_metrics2['f1']
        print('Performance on test set: precision: {:5.2f} | recall: {:5.2f} | accuarcy: {:5.4f} | f1: {:5.2f}'.format(precision, recall,  accuarcy,f1))
