MICRO_TARGET_TICKERS = [
    "005930", "000660", "373220", "207940", "005380", "005490", "000270", "105560", "068270", "035420", # 1-10
    "005935", "051910", "035720", "000660", "006400", "012330", "028260", "105560", "055550", "003550", # 11-20
    "032830", "015760", "000810", "033780", "009150", "086790", "011780", "018260", "010130", "034730", # 21-30
    "011200", "000100", "034220", "010950", "051900", "012450", "009830", "003670", "009540", "047050", # 31-40
    "086280", "017670", "011070", "042700", "011170", "005830", "267250", "036570", "024110", "004020", # 41-50
    "006800", "000720", "005940", "002790", "010620", "032640", "071050", "039490", "000120", "000150", # 51-60
    "004990", "021240", "001450", "007070", "008770", "003410", "138040", "003490", "251270", "011790", # 61-70
    "096770", "016360", "004170", "023530", "000080", "078930", "001040", "006360", "005090", "001470", # 71-80
    "010120", "004370", "069500", "005250", "000210", "005440", "009420", "010140", "139130", "004800", # 81-90
    "064350", "000990", "014680", "001800", "052690", "003240", "007310", "006040", "003000", "005180"  # 91-100
]

MACRO_TARGET_TICKERS = [
    # --- 시장 대표 지수 (국내) ---
    "069500",  # KODEX 200 (코스피 200 대용)
    "229200",  # KODEX 코스닥150 (코스닥 150 대용)
    "005930",  # 삼성전자 (시장 전체 심리 및 반도체 업황 지표)

    # --- 글로벌 주요 지수 (해외) ---
    "133690",  # TIGER 미국나스닥100 (미국 기술주 흐름)
    "143850",  # TIGER 미국S&P500 (글로벌 증시 표준)
    "192090",  # TIGER 차이나CSI300 (중국 경기 및 수출 환경)
    "241180",  # TIGER 일본니케이225 (아시아 자금 흐름)

    # --- 환율 및 안전자산 ---
    "261240",  # KODEX 미국달러선물 (원/달러 환율 변동성)
    "132030",  # TIGER 금은선물(H) (안전선호 심리 및 인플레이션)

    # --- 원자재 및 에너지 ---
    "261220",  # KODEX WTI원유선물(H) (에너지 가격 및 물가)
    "138910",  # TIGER 구리실물 (실물 경기 회복 선행 지표)

    # --- 물류 및 공포지수 ---
    "011200",  # HMM (해운 운임 및 글로벌 물류 지표 대용)
    "271050",  # KODEX 미국S&P500VIX선물(H) (시장 변동성 및 공포지수)
]

INPUT_WINDOW = 20 #스케일링을 할때 기준 길이
LABEL_WINDOW = 20 #라벨링시 활용할 윈도우 크기
INCLUDE_DOW = True
INCLUDE_MONTH = False

TRAIN_RATIO = 0.5
VAL_RATIO = 0.47

#모델별 아키텍처 및 학습 하이퍼 파라미터
DEFAULT_PARAMS = {
    'TNC_pretrain': {
        'arch_params' : {
            'window_size': 128,      # TNC 데이터셋 분할용 (이 값이 seq_len과 동일해야 함)
            'mc_sample_size': 10,    # Positive 샘플링 범위
            'input_dim': 38,          # 피처 개수 (예: OHLCV)
            'seq_len': 128,          # 입력 시퀀스 길이
            'patch_len': 4,         # 패치 길이
            'stride': 8,             # 패치 스트라이드
            'd_model': 64,           # 트랜스포머 차원
            'n_heads': 4,
            'n_layers': 3,
            'z_dim': 32,             # 최종 압축 임베딩 차원
            'disc_hidden_dim': 64    # 판별자 은닉층 차원
        },
        'train_params' : {
            'batch_size': 256,
            'epochs': 20,
            'lr': 1e-3
        }
    },
    'TFC_Pretrain': {
        'arch_params': {
            'ts_length': 16,
            'num_features': 38,
            'num_layers': 2,
            "nhead" : 2,
            "feat_dim" : 128
        },
        'train_params': {
            'batch_size': 1024,
            'epochs': 60,
            'lr': 1e-4,
            'temperature': 0.2
        }
    },
    'TFC_Head': {
        'arch_params': {
            'ts_length': 16,
            'head_in_features': 256,
            'head_out_classes': 3
        },
        'train_params': {
            'batch_size': 64,
            'epochs': 10,
            'lr': 1e-4
        },
        'embed_info': {
            'name': 'TFC_Embed',
            'weights_path': 'TFC.pth',
            'arch_params': {
                'ts_length': 32,
                'num_features': 38,
                'num_layers': 2,
                "nhead" : 2,
                "feat_dim" : 128
            }
        }
    },
    'XGB': {
        'objective': 'multi:softprob',
        'num_class': 5,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',      # XGBoost GPU 사용을 위한 트리 구조
        'device': 'cuda',
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    },
    'LGBM': {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'device_type': 'gpu',       # LightGBM GPU 사용
        'boosting_type': 'gbdt',
        'class_weight': 'balanced',
        'max_depth': 4,
        'num_leaves': 15,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 30,
        'random_state': 42,
        'verbose': -1,
    }
}
