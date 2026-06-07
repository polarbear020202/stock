import config
from data_loader import GenerateDataLoader, TNCDataset, LabeledTimeSeriesDataset, TimeSeriesDataset
from TNC import TNC, TNCTrainer
from TFC import TFC, TFCTrainer, NTXentLoss, target_classifier, HeadTrainer
import torch
import torch.nn as nn

class UnifiedTimeSeriesTrainer:
    def __init__(
        self,
        arch_name: str,
        df_dict: dict,
        arch_params: dict = None,
        train_params: dict = None,
        embed_info: dict = None
    ):
        self.arch_name = arch_name
        self.arch_params = arch_params
        self.train_params = train_params
        self.df_dict = df_dict
        self.embed_info = embed_info or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, tune=False):
        print(f"[{self.arch_name}] 학습을 시작합니다. (Device: {self.device})")

        if self.arch_name == 'TFC_SSL':
            self._train_tfc_ssl()
        elif self.arch_name == 'TFC_Head':
            self._train_tfc_head()
        elif self.arch_name == 'TNC_SSL':          # <--- 추가된 부분
            self._train_tnc_ssl()
        else:
            raise ValueError(f"지원하지 않는 아키텍처입니다: {self.arch_name}")

    # ===== [TNC SSL 추가 구현부] =====
    def _train_tnc_ssl(self):
        # 1. 파라미터 추출
        batch_size = self.train_params.get('batch_size', 1024)
        epochs = self.train_params.get('epochs', 10)
        lr = self.train_params.get('lr', 1e-3)
        window_size = self.arch_params.get('window_size', 100)
        mc_sample_size = self.arch_params.get('mc_sample_size', 10)

        # 2. 데이터 로더 준비 (이전에 만든 TNCDataset 사용)
        get_dataloaders = GenerateDataLoader(batch_size=batch_size, gap=window_size)
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_class=TNCDataset,
            dfs_dict=self.df_dict,
            window_size=window_size,
            mc_sample_size=mc_sample_size
        )

        # 3. 모델, 손실 함수, 옵티마이저 초기화
        # arch_params에 input_dim, hidden_dim, z_dim이 포함되어야 함
        model = TNC(
            input_dim=self.arch_params['input_dim'],
            hidden_dim=self.arch_params['hidden_dim'],
            z_dim=self.arch_params['z_dim']
        ).to(self.device)

        criterion = nn.BCELoss() # TNC는 이진 교차 엔트로피 사용
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 4. Trainer 생성 및 학습 실행
        trainer = TNCTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            epochs=epochs
        )
        trainer.fit()

    def _train_tfc_ssl(self):
        # 1. 파라미터 추출
        batch_size = self.train_params.get('batch_size', 1024)
        epochs = self.train_params.get('epochs', 10)
        lr = self.train_params.get('lr', 1e-3)
        temperature = self.train_params.get('temperature', 0.2)

        # 2. 데이터 로더 준비
        get_dataloaders = GenerateDataLoader(batch_size=batch_size, gap=self.arch_params['ts_length'])
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_class=TimeSeriesDataset,
            dfs_dict=self.df_dict,
            window_size=self.arch_params['ts_length']
        )

        # 3. 모델, 손실 함수, 옵티마이저 초기화
        model = TFC(**self.arch_params).to(self.device)

        criterion = NTXentLoss(
            device=self.device,
            temperature=temperature,
            use_cosine_similarity=True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 4. Trainer 생성 및 학습 실행
        trainer = TFCTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            epochs=epochs
        )
        trainer.fit()

    def _train_tfc_head(self):
        # 1. 파라미터 추출
        ts_length = self.arch_params['ts_length']
        head_in_features = self.arch_params['head_in_features']
        head_out_classes = self.arch_params['head_out_classes']

        batch_size = self.train_params['batch_size']
        epochs = self.train_params['epochs']
        lr = self.train_params['lr']

        # 임베딩 모델 파라미터
        embed_num_features = self.embed_info['arch_params']['num_features']
        embed_weights_path = self.embed_info['weights_path']

        # 2. 데이터 로더 준비
        get_dataloaders = GenerateDataLoader(batch_size=batch_size, gap=ts_length)
        labeled_train_loader, labeled_val_loader, labeled_test_loader = get_dataloaders(
            dataset_class=LabeledTimeSeriesDataset,
            dfs_dict=self.df_dict, # head 학습 시에는 labeled_df_dict가 들어와야 함
            window_size=ts_length
        )

        # 3. 임베딩 모델 로드 및 동결
        embed_model = TFC(**self.embed_info['arch_params']).to(self.device)
        embed_model.load_state_dict(torch.load(embed_weights_path, map_location=self.device))
        embed_model.eval()

        # 4. Head 모델, 손실 함수, 옵티마이저 초기화
        head_model = target_classifier(head_in_features, head_out_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(head_model.parameters(), lr=lr)

        # 5. Trainer 생성 및 학습 실행
        trainer = HeadTrainer(
            embed_model=embed_model,
            model=head_model,
            train_loader=labeled_train_loader,
            val_loader=labeled_val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            epochs=epochs,
            num_classes=head_out_classes
        )
        trainer.fit()
