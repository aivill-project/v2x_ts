from v2x import V2XData, V2XDataLabeled
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import sklearn.metrics as skm
from tsai.all import *
from fastai.callback.wandb import *
from modules.RunTSAI import RunTSAI
from tqdm import tqdm
# import wandb
# my_setup(wandb)
# wandb.login()

config_default = AttrDict(
    batch_tfms = TSStandardize(by_sample=True),
    arch_config = {},
    architecture = LSTM, # [LSTM, LSTMPlus, LSTM_FCN, LSTM_FCNPlus, MLSTM_FCN, MLSTM_FCNPlus]
    lr = 1e-3,
    n_epochs = 20,
)


data_path = 'data/labeled/220801/C'
conditions = ['A', 'D', 'F', 'N']
v2xdata_A = V2XDataLabeled(data_path, condition=conditions[0])

# X_sum, y_sum = empty numpy array
X_sum = np.empty((0, 50, 1))
y_sum = np.empty((0, 1))
for idx in tqdm(range(0, 10)):
    X, y, splits, df = v2xdata_A[idx]
    X_sum = np.concatenate((X_sum, X), axis=0) if X_sum.size else X
    y_sum = np.concatenate((y_sum, y), axis=0) if y_sum.size else y
splits = V2XDataLabeled.get_splits(None,X_sum)
X_sum.shape, y_sum.shape, splits

learner = RunTSAI.multivariate_classification(X_sum, y_sum[:, 0], splits, config=config_default)
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix()