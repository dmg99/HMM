from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import HMM

path = '../data/train_clean.csv'


def read_signal(path):
  temp_train = pd.read_csv(path)
  signal = np.array(temp_train['signal'])
  if 'open_channels' in temp_train.columns:
    chan = np.array(temp_train['open_channels'])
    return signal, chan
  else:
    return signal


fm = 10000
signal, chan = read_signal(path)

inf_train = [int(20e5), int(45e5)]
limit_train = [int(25e5), int(50e5)]
inf_pred = [int(20e5), int(45e5)]
limit_pred = [int(25e5), int(50e5)]

signal_bona = np.concatenate([signal[x:y] for x,y in zip(inf_train,limit_train)]) 
chan_bona = np.concatenate([chan[x:y] for x,y in zip(inf_train,limit_train)]) 
signal_pred = np.concatenate([signal[x:y] for x,y in zip(inf_pred,limit_pred)]) 
signal_pred_vec = HMM.numpy2vec(signal_pred)
chan_pred = np.concatenate([chan[x:y] for x,y in zip(inf_pred,limit_pred)]) 
signal_vec = HMM.numpy2vec(signal_bona)
chan_vec = HMM.numpy2vec_int(chan_bona)

print(np.unique(chan_bona))

'''
model = HMM.HMM(3)
model.switch_pdf_constant()
model.EM_maximization(signal_vec, chan_vec, 0.0001)
model.set_decode_iter_print(100000)
pred = np.array(model.decode(signal_vec))
print('MODEL TONI:')
print(f1_score(chan[:limit], pred%2))
'''

'''
model = HMM.HiddenHMM_comp(11, 8)
model.switch_state_training()
#model.switch_pdf_constant()
model.fit_model_params_from_truth(signal_vec, chan_vec)
model.EM_maximization(signal_vec, chan_vec, 0.001, 1500)
pred = np.array(model.decode(signal_pred_vec))
print(f1_score(chan_pred, pred[0], average=None))
'''
model = HMM.HiddenHMM(11, 5)
model.switch_state_training()
#model.switch_pdf_constant()
model.fit_model_params_from_truth(signal_vec, chan_vec)
model.EM_maximization(signal_vec, chan_vec, 0.001, 1500)
pred = np.array(model.decode(signal_pred_vec))
print(f1_score(chan_pred, pred[0], average=None))

'''

signal_vec = HMM.numpy2vec(np.concatenate((signal[2000000:2500000], signal[4500000:])))
chan_vec = HMM.numpy2vec_int(np.concatenate((chan[2000000:2500000], chan[4500000:])))

model = HMM.HiddenHMM(11, 2)
#model.switch_pdf_constant()
model.fit_model_params_from_truth(signal_vec, chan_vec)
model.EM_maximization(signal_vec, chan_vec, 0.0001, 300)
pred = np.array(model.decode(signal_vec))
print(np.unique(pred))
print(f1_score(np.concatenate((chan[2000000:2500000], chan[4500000:])), pred[0], average='macro'))
'''
