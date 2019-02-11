
import sklearn.preprocessing as pre, sys, pyemma as py, numpy as np, matplotlib.pyplot as plt, glob, pickle, os, pickle
sys.path.append('/home/weichen9/scratch-midway2/HDE_torch/hde/')

sys.path.append('/home/kengyangyao/Dropbox/temp_Linux/temp_research_proj/hde_vac/')
sys.path.append('/home/kengyangyao/Dropbox/temp_Linux/temp_research_proj/HDE_torch/hde/')

from hde import HDE, analysis
from hde_torch import *
lag_time = 2000
dt = 0.2  # unit: ns

index = int(sys.argv[1])
mode = 'keras'

try:
    combined_data_s = [np.load('/home/weichen9/scratch-midway2/HDE_data/WW_domain/pair_dis_ca_s_%d.npy' % item) for item in range(2)]
    tica_coords = pickle.load(open('/home/weichen9/scratch-midway2/HDE_data/WW_domain/tica_coords.pkl', 'rb'))
except:
    combined_data_s = [np.load('/home/kengyangyao/data/HDE_data/WW_domain/pair_dis_ca_s_%d.npy' % item) for item in range(2)]
    tica_coords = pickle.load(open('/home/kengyangyao/data/HDE_data/WW_domain/tica_coords.pkl', 'rb'))

hde_skip = 4
data = [item[::hde_skip].astype(np.float64) for item in combined_data_s]

if mode == 'torch':
    hde_t = HDE_torch(595, n_out=2, lag=lag_time // hde_skip, lr=0.01, batch_normalization=True,
                    koopman_weighting_version=-1, batch_size=200000, epochs=30, 
                    validation_split=0.2,
                    ).cuda()

    hde_t.fit(data)
    hde_t.save_to('temp_torch_ww_%02d.pth' % index)
    hde_timescales = hde_t.timescales_ * hde_skip
    print(hde_timescales)  
    hde_coords = hde_t.transform(combined_data_s, batch_size=10000)
elif mode == 'keras':
    from keras.callbacks import EarlyStopping
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=50, verbose=1, mode='min', restore_best_weights=True)
    hde = HDE(595, n_components=2, lag_time=lag_time // hde_skip, dropout_rate=0, batch_size=200000, n_epochs=30,
            validation_split=0., batch_normalization=True, learning_rate=0.01, hidden_size=100,
            callbacks=[earlyStopping])
    hde.fit(data)
    pickle.dump(hde, open('temp_keras_ww_%02d.pkl' % index, 'wb') )
    hde_timescales = hde.timescales_ * hde_skip
    print(hde_timescales)
    hde_coords = [hde.transform(item) for item in combined_data_s]

np.save('hde_coords_ww_%02d' % index, hde_coords)
np.save('hde_timescales_ww_%02d' % index, hde_timescales)
