import data
import config
import labeler

import pickle
from pathlib import Path

df_dict = data.KRXDataCollector(years=20, delay=0.4)(config.MICRO_TARGET_TICKERS, config.MACRO_TARGET_TICKERS)
labeled_df_dict = labeler.DfDictIterator()(df_dict, labeler.labeler(window=config.LABEL_WINDOW, ret_col='log_ret', min_return=0.1, max_time=40 ))

data_to_save = {
    'raw_data': df_dict,
    'labeled_data': labeled_df_dict
}

folder_path = Path('./saved_data')
folder_path.mkdir(parents=True, exist_ok=True)
file_path = folder_path / 'my_financial_data.pkl'

#저장하는 코드
with open('my_financial_data.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

#불러오는 코드
with open('my_financial_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# 원래대로 다시 할당
df_dict = loaded_data['raw_data']
labeled_df_dict = loaded_data['labeled_data']