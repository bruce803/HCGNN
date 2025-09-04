import numpy as np
import torch
import argparse

def compute_ext(data, jicha_windows, mode):
    if type(data) == torch.Tensor:
        b, t, N = data.size() # x.shape=batch,*,N
        data= data.view(1, -1, N).squeeze().detach().cpu().numpy()
    if mode=='重叠窗口':
        '''*******'''
    elif mode=="不重叠窗口":
        n=b*t//jicha_windows
        data=np.array_split(data,n,axis=0)
        max=list(map(lambda i:i.max(axis=0),data))
        min=list(map(lambda i:i.min(axis=0),data))
        jicha=[max[i]-min[i] for i in range(len(max))]
        # max_index=list(map(lambda i:i.argmax(axis=0),data))
        # min_index=list(map(lambda i:i.argmin(axis=0),data))
        # jicha=[jicha[i]*(-1) if max_index[i]>min_index[i] for i in range(len(jicha))]
        jicha=torch.tensor(np.array(jicha))
        return jicha

def moving_average(x,w): #
    return np.convolve(x, np.ones(w), 'valid') / w

def MA(data,MA_windows): #计算移动平均
    if type(data) == torch.Tensor:
        b, t, N = data.size() # x.shape=batch,*,N
        data= data.view(1, -1, N).squeeze().detach().cpu().numpy()

        ma=np.empty((b*t-MA_windows+1,N))
        for k in range(N):
            ma_x=moving_average(data[:,k],MA_windows).reshape(b*t-MA_windows+1,1)
            ma=np.append(ma,ma_x,axis=1)
        ma=ma[:,N:]
        ma=torch.tensor(ma,dtype=torch.float) #torch.tensor
    elif type(data) == np.ndarray:
        t, N = data.shape
        ma = np.empty((t - MA_windows + 1, N))
        for k in range(N):
            ma_x = moving_average(data[:, k], MA_windows).reshape( t - MA_windows + 1, 1)
            ma = np.append(ma, ma_x, axis=1)
        ma = ma[:, N:]
    return ma

def masked_MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mask = (v == 0)
    percentage = np.abs(v_ - v) / np.abs(v)
    if np.any(mask):
        masked_array = np.ma.masked_array(percentage, mask=mask)  # mask the dividing-zero as invalid
        result = masked_array.mean(axis=axis)
        if isinstance(result, np.ma.MaskedArray):
            return result.filled(np.nan)
        else:
            return result
    return np.mean(percentage, axis).astype(np.float64)


def MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mape = (np.abs(v_ - v) / np.abs(v)+1e-5).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)


def RMSE(v, v_, axis=None):
    '''
    Mea00n squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.
    '''
    np.seterr(divide='ignore', invalid='ignore')

    return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64)


def MAE(v, v_, axis=None):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v), axis).astype(np.float64)

def TP(output, target):
    assert len(output) == len(target), "output len: {} != target len: {}".format(len(output), len(target))
    base = np.ones(output.shape)
    a = np.array(target == output).astype(int)  # TP+TN
    b = np.array(target == base).astype(int)    # TP+FN
    a[a == 0] = -1
    tp = np.sum(a == b)
    return tp
def FP(output, target):
    assert len(output) == len(target), "output len: {} != target len: {}".format(len(output), len(target))
    base = np.ones(output.shape)
    a = np.array(target != output).astype(int)
    a[a == 0] = -1
    b = np.array(output == base).astype(int)
    fp = np.sum(a == b)
    return fp
def FN(output, target):
    assert len(output) == len(target), "output len: {} != target len: {}".format(len(output), len(target))
    base = np.ones(output.shape)
    a = np.array(target != output).astype(int)
    a[a == 0] = -1
    b = np.array(target == base).astype(int)
    fp = np.sum(a == b)
    return fp
def TN(output, target):
    assert len(output) == len(target), "output len: {} != target len: {}".format(len(output), len(target))
    base = np.zeros(output.shape)
    a = np.array(target == output).astype(int)
    a[a == 0] = -1
    b = np.array(output == base).astype(int)
    fp = np.sum(a == b)
    return fp
def precision(output, target, axis=None):
    if type(output) != torch.Tensor:
        tp= TP(output, target)
        fp=FP(output, target)
        return tp/(tp+fp)
def recall(output, target, axis=None):
    if type(output) != torch.Tensor:
        tp = TP(output, target)
        fn = FN(output, target)
        return tp/(tp + fn )
def accuarcy(output, target, axis=None):
    if type(output) != torch.Tensor:
        tp = TP(output, target)
        fp = FP(output, target)
        fn = FN(output, target)
        tn = TN(output, target)
        return (tp + tn) / (tp + fp + tn + fn)
def f1(output, target, axis=None):
    if type(output) != torch.Tensor:
        p=precision(output, target, axis)
        r=recall(output, target, axis)
        return 2*p*r/(p+r)

def evaluate(Regression, Classification,y, y_hat, by_step=False, by_node=False):
    '''
    :param y: array in shape of [count, time_step, node].
    :param y_hat: in same shape with y.
    :param by_step: evaluate by time_step dim.
    :param by_node: evaluate by node dim.
    :return: array of mape, mae and rmse.
    '''

    if Regression:
        if not by_step and not by_node:
            return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat)
        if by_step and by_node:
            return MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0)
        if by_step:
            return MAPE(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2))
        if by_node:
            return MAPE(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1))
    if Classification:
        if not by_step and not by_node:
            return precision(y_hat,y), recall(y_hat,y), accuarcy(y, y_hat), f1(y,y_hat)
        if by_step and by_node:
            return precision(y_hat,y, axis=0), recall(y_hat,y, axis=0), accuarcy(y, y_hat, axis=0), f1(y,y_hat, axis=0)
        if by_step:
            return precision(y_hat,y, axis=(0, 2)), recall(y_hat,y, axis=(0, 2)), accuarcy(y, y_hat, axis=(0, 2)), f1(y,y_hat, axis=(0, 2))
        if by_node:
            return precision(y_hat,y, axis=(0, 1)), recall(y_hat,y, axis=(0, 1)), accuarcy(y, y_hat, axis=(0, 1)), f1(y,y_hat, axis=(0, 1))