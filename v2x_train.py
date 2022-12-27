from modules.v2x import V2XData,V2XDataLabeled
import sklearn.metrics as skm
from sklearn.utils import shuffle
from tsai.all import *
from fastai.callback.wandb import *
from modules.RunTSAI import RunTSAI
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--class_name", type=str, default="False", help="turn, lane, speed, hazard")
args = parser.parse_args()

conditions = ['A', 'D', 'F', 'N']
dates_range = ['220801', '220802', '220803', '220804', '220805', '220806', '220807', '220808', '220809', '220810', '220811', '220812', '220813', '220814', '220815', '220816', '220817', '220818', '220819', '220820', '220821', '220822', '220823', '220824', '220825', '220826', '220827', '220828', '220829', '220830', '220831']


config_default = AttrDict(
    batch_tfms = TSStandardize(by_sample=True),
    arch_config = {},
    architecture = MLSTM_FCNPlus, # [LSTM, LSTMPlus, LSTM_FCN, LSTM_FCNPlus, MLSTM_FCN, MLSTM_FCNPlus]
    lr = 1e-3,
    n_epochs = 20,
)

# combine all the data
v2x_data_x = np.load("data/X_sum_all.npy", allow_pickle=True)
v2x_data_y = np.load("data/y_sum_all.npy", allow_pickle=True)
y_turn, y_lane, y_speed, y_hazard = v2x_data_y[:, 0], v2x_data_y[:, 1], v2x_data_y[:, 2], v2x_data_y[:, 3]
X = v2x_data_x

splits_turn = get_splits(v2x_data_y[:, 0], valid_size=0.2, test_size=0.2, random_state=42)
splits_lane = get_splits(v2x_data_y[:, 1], valid_size=0.2, test_size=0.2, random_state=42)
splits_speed = get_splits(v2x_data_y[:, 2], valid_size=0.2, test_size=0.2, random_state=42)
splits_hazard = get_splits(v2x_data_y[:, 3], valid_size=0.2, test_size=0.2, random_state=42)
print(splits_turn, splits_lane, splits_speed, splits_hazard)


learner_turn = RunTSAI.multivariate_classification(X, y_turn, splits_turn, config)
curr_time = datetime.now().strftime("%Y%m%d_%H%M")
learner_turn.save_all(path=f"models/turn_{curr_time}", dls_fname='dls_turn', model_fname='model_turn_MLSTM_FCNPlus', learner_fname='learner_turn')

if __name__ == "__main__":
    if args.class_name == "turn":
        learner_turn = RunTSAI.multivariate_classification(X, y_turn, splits_turn, config)
        curr_time = datetime.now().strftime("%Y%m%d_%H%M")
        learner_turn.save_all(path=f"models/turn_{curr_time}", dls_fname='dls_turn', model_fname='model_turn_MLSTM_FCNPlus', learner_fname='learner_turn')
    elif args.class_name == "lane":
        learner_lane = RunTSAI.multivariate_classification(X, y_lane, splits_lane, config)
        curr_time = datetime.now().strftime("%Y%m%d_%H%M")
        learner_lane.save_all(path=f"models/lane_{curr_time}", dls_fname='dls_lane', model_fname='model_lane_MLSTM_FCNPlus', learner_fname='learner_lane')
    elif args.class_name == "speed":
        learner_speed = RunTSAI.multivariate_classification(X, y_speed, splits_speed, config)
        curr_time = datetime.now().strftime("%Y%m%d_%H%M")
        learner_speed.save_all(path=f"models/speed_{curr_time}", dls_fname='dls_speed', model_fname='model_speed_MLSTM_FCNPlus', learner_fname='learner_speed')
    elif args.class_name == "hazard":
        learner_hazard = RunTSAI.multivariate_classification(X, y_hazard, splits_hazard, config)
        curr_time = datetime.now().strftime("%Y%m%d_%H%M")
        learner_hazard.save_all(path=f"models/hazard_{curr_time}", dls_fname='dls_hazard', model_fname='model_hazard_MLSTM_FCNPlus', learner_fname='learner_hazard')
    else:
        print("Please specify the class name to train.[turn, lane, speed, hazard]")