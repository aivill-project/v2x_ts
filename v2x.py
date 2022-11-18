from glob import glob
from tqdm import tqdm
import os
import pandas as pd
from tsai.all import *
from tsai.models.MINIROCKET import *
from fastai.torch_core import default_device
from fastai.metrics import accuracy
from fastai.callback.tracker import ReduceLROnPlateau
from tsai.data.all import *
from tsai.learner import *
from sklearn.model_selection import train_test_split
from tsai.basics import *
from tsai.data.external import *
from tsai.data.preprocessing import *
from torch.utils.data import DataLoader, Dataset
import numpy as np

DROP_COLS = ['VEHICLE_CLASS', 'VEHICLE_TYPE', 'Turn', 'Change', 'Speed', 'ISSUE_DATE', 'X', 'Y']
TARGET_COLS = ["Hazard"]
INDEX_COL = "ISSUE_DATE"


class V2XData():
    def __init__(self, dataset_path = None, dataset_files: list = None, drop_cols: list = None, 
                 target_cols: list = None, test_size: float = 0.5, random_state: int = None,
                 hazard_thr: int = 1):
        self.data_path = dataset_path
        if dataset_files:
            self.dataset_files = dataset_files.copy()
        else:
            self.dataset_files = sorted(glob.glob(os.path.join(self.data_path, "*.csv")))
        
        if not drop_cols:
            self.drop_cols = DROP_COLS.copy()    
        else:
            self.drop_cols = drop_cols
        
        if not target_cols:
            self.target_cols = TARGET_COLS.copy()
        else:
            self.target_cols = target_cols
        
        self.test_size = test_size
        self.random_state = random_state
        self.files_num = len(self.dataset_files)
        
        if hazard_thr < 1:
            assert hazard_thr > 0, "hazard_thr must be greater than 0"
        else: 
            self.hazard_thr = hazard_thr
        
        print(f"loaded {len(self.dataset_files)} files")
        print(self.dataset_files[:5], "...")
        
    def __getitem__(self, index, print_size=True):
        df = pd.read_csv(self.dataset_files[index]).drop(labels = self.drop_cols, axis=1)
        df.dropna(0, inplace = True)
        if print_size: print(f'df[{index}] shape: {df.shape}')
        # print(df.info())
        df_filtered = pd.crosstab(df['scene'], df['Hazard'])
        df = df.groupby(df['scene']).filter(lambda x: len(x) == 10)
        df.reset_index(drop=True, inplace=True)
        change_list = list(df_filtered[df_filtered[True] >= self.hazard_thr].index)
        df.loc[df['scene'].isin(change_list), 'Hazard'] = True
        
        y = df['Hazard'].iloc[::10]
        y = y.astype(int)
        y = y.to_numpy()
        
        X = df.groupby(df["scene"]).apply(lambda x: x.drop(["scene", "Hazard"], axis=1).values)
        X = np.array(X.tolist())
        X = np.array([scene.transpose() for scene in X])
        
        splits = self.get_splits(X, test_size = self.test_size, random_state = self.random_state)
        
        return X, y, splits, df
    
    def get_splits(self, X, test_size: float = 0.5, random_state: int = None):
        X_train, X_valid = train_test_split(X, test_size=test_size, random_state=random_state)
        splits = get_predefined_splits(X_train, X_valid)
        return splits
    
    @staticmethod
    def get_data_info(X, y, splits = None):
        print('Dataset Info is...')
        print(f'X shape: {X.shape}, y shape: {y.shape}') 
        if(splits):
            print(f'splits: (train: (#{len(splits[0])})({splits[0][0]}, ...)) (test: (#{len(splits[1])})({splits[1][0]}, ...))')
        print(f'# True in y: {np.unique(y, return_counts=True)}')
        print('Dataset Info is done.')
    
    def get_all_item(self, is_test = False, print_size=True):
        X_sum, y_sum = [], []
        df_sum = pd.DataFrame()
        files_num = len(self.dataset_files)
        if is_test == True:
            files_num = int(files_num - 0.2*files_num)
        print(f'files index: 0 ~ {files_num}')
        
        for idx in tqdm(range(files_num)):
            X, y, _, df = self.__getitem__(idx, print_size=False)
            X_sum.append(X)
            y_sum.append(y)
            df_sum = pd.concat([df_sum, df])
            
            
        X_sum = np.concatenate(X_sum)
        y_sum = np.concatenate(y_sum)
        df_sum.reset_index(drop=True, inplace=True)
        splits = self.get_splits(X_sum, test_size = self.test_size, random_state = self.random_state)
        print(f'X_sum shape: {X_sum.shape}, y_sum shape: {y_sum.shape}')
        return X_sum, y_sum, splits, df_sum
    