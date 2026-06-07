import torch
import torch.nn as nn

from tqdm import tqdm

class RevIN(nn.Module):
    def __init__(self, num_features: int = 1, eps=1e-5, affine=True):
        """
        단변량일 경우 num_features는 1로 설정합니다.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        # 모델이 스스로 스케일과 시프트를 학습할 수 있게 해주는 파라미터
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'calc_and_norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'norm':
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        # x shape: [batch_size, sequence_length, num_features]
        # 시간축(dim=1)을 기준으로 평균과 분산을 계산하여 내부 변수로 저장
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        # 저장된 평균과 분산으로 데이터 정규화
        x = x - self.mean
        x = x / self.stdev

        # 학습 가능한 파라미터(Affine) 적용
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        # 정규화의 정확히 역순으로 연산 (Denormalization)
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)

        x = x * self.stdev
        x = x + self.mean
        return x

class TNCEncoder(nn.Module):
    def __init__(self, input_dim, seq_len, patch_len, stride, d_model=64, n_heads=4, n_layers=3, z_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # 0. RevIN 모듈 추가 (입력 피처 수에 맞게 초기화)
        self.revin = RevIN(num_features=input_dim, affine=True)
        
        # 1. 패치 개수 계산
        self.num_patches = int((seq_len - patch_len) / stride) + 1
        
        # 2. Patch Projection
        self.patch_proj = nn.Linear(patch_len, d_model)
        
        # 3. Positional Encoding
        self.W_pos = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            activation='gelu',
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 5. 최종 Representation (z) 생성용 FC Layer
        self.flatten_dim = input_dim * self.num_patches * d_model
        self.fc = nn.Linear(self.flatten_dim, z_dim)

    def forward(self, x):
        """
        x shape: (Batch_size, Seq_len, Input_dim) -> (B, L, M)
        """
        # [Step 0] RevIN을 통한 인스턴스 정규화 (핵심 추가 부분)
        # 패치화나 채널 분리 이전에, 시계열 전체 길이(L)에 대해 정규화를 먼저 수행합니다.
        x = self.revin(x, mode='calc_and_norm')

        B, L, M = x.shape
        
        # [Step 1] Channel Independence (채널 독립성)
        # (B, L, M) -> (B, M, L) -> (B * M, L)
        x = x.permute(0, 2, 1).reshape(B * M, L)
        
        # [Step 2] Patching (패치화)
        # (B * M, L) -> (B * M, num_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # [Step 3] Patch Embedding & Positional Encoding
        # (B * M, num_patches, patch_len) -> (B * M, num_patches, d_model)
        x = self.patch_proj(x) 
        x = x + self.W_pos
        
        # [Step 4] Transformer Encoder
        # (B * M, num_patches, d_model)
        x = self.transformer_encoder(x)
        
        # [Step 5] 형태 복구 및 TNC용 최종 벡터 생성
        # (B * M, num_patches, d_model) -> (B, M, num_patches, d_model)
        x = x.view(B, M, self.num_patches, self.d_model)
        
        # (B, M * num_patches * d_model)
        x = x.reshape(B, -1)
        
        # (B, z_dim)
        z = self.fc(x)
        
        return z
    
class TNCDiscriminator(nn.Module):
    def __init__(self, z_dim, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )

    def forward(self, z_t, z_p_or_n):
        feat = torch.cat([z_t, z_p_or_n], dim=-1)
        return self.model(feat)

class TNC(nn.Module):
    def __init__(
        self, 
        # --- Encoder 파라미터 ---
        input_dim, 
        seq_len, 
        patch_len, 
        stride, 
        d_model=64, 
        n_heads=4, 
        n_layers=3, 
        z_dim=32,
        # --- Discriminator 파라미터 ---
        disc_hidden_dim=64,
        **kwargs
    ):
        super().__init__()
        
        # 1. 인코더 초기화 (모든 트랜스포머 및 패치 파라미터 전달)
        self.encoder = TNCEncoder(
            input_dim=input_dim,
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            z_dim=z_dim
        ) 
        
        # 2. 판별자 초기화 (z_dim과 판별자 전용 은닉층 차원 전달)
        self.discriminator = TNCDiscriminator(
            z_dim=z_dim, 
            hidden_dim=disc_hidden_dim
        ) 

    def forward(self, anchor, pos, neg):
        # 1. 인코딩: 각각의 윈도우를 Patch-Transformer를 거쳐 z 벡터로 압축
        z_anchor = self.encoder(anchor)
        z_pos = self.encoder(pos)
        z_neg = self.encoder(neg)
        
        # 2. 판별자 통과: 기준(anchor) 대비 이웃(pos) 및 비이웃(neg) 확률 계산
        prob_pos = self.discriminator(z_anchor, z_pos).squeeze(-1)
        prob_neg = self.discriminator(z_anchor, z_neg).squeeze(-1)
        
        return prob_pos, prob_neg

class TNCTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, epochs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for anchor, positive, negative in progress_bar:
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            self.optimizer.zero_grad()

            # 모델의 Forward에서 확률값 계산
            prob_pos, prob_neg = self.model(anchor, positive, negative)

            # 라벨 생성: Positive는 1, Negative는 0
            label_pos = torch.ones_like(prob_pos)
            label_neg = torch.zeros_like(prob_neg)

            # 손실 계산
            loss_pos = self.criterion(prob_pos, label_pos)
            loss_neg = self.criterion(prob_neg, label_neg)
            loss = loss_pos + loss_neg

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for anchor, positive, negative in progress_bar:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                prob_pos, prob_neg = self.model(anchor, positive, negative)

                label_pos = torch.ones_like(prob_pos)
                label_neg = torch.zeros_like(prob_neg)

                loss_pos = self.criterion(prob_pos, label_pos)
                loss_neg = self.criterion(prob_neg, label_neg)
                loss = loss_pos + loss_neg

                total_loss += loss.item()
                progress_bar.set_postfix({'val_loss': f"{loss.item():.4f}"})

        return total_loss / len(self.val_loader)

    def fit(self):
        max_val_loss = float('inf')
        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch [{epoch}/{self.epochs}]")

            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            if val_loss < max_val_loss:
                max_val_loss = val_loss
                torch.save(self.model.state_dict(), 'TNC.pth')
                print("Model saved!")

            print(f"-> Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")