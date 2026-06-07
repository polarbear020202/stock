import torch
import torch.nn as nn

from tqdm import tqdm

class TNCEncoder(nn.Module):
    def __init__(self, input_dim, seq_len, patch_len, stride, d_model=64, n_heads=4, n_layers=3, z_dim=32):
        """
        Args:
            input_dim: 시계열 피처(채널) 개수 (M)
            seq_len: 입력 시계열의 전체 길이 (L)
            patch_len: 각 패치의 길이 (P)
            stride: 패치 간 이동 간격 (S)
            d_model: 트랜스포머 모델 차원
            n_heads: 트랜스포머 Multi-head 개수
            n_layers: 트랜스포머 인코더 레이어 수
            z_dim: TNC Discriminator로 넘어갈 최종 임베딩 벡터 차원
        """
        super().__init__()
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # 1. 패치 개수 계산 공식: (L - P) / S + 1
        self.num_patches = int((seq_len - patch_len) / stride) + 1
        
        # 2. Patch Projection (Shared Encoder의 진입점)
        # 패치 내의 값들을 d_model 차원으로 임베딩 (가중치 공유)
        self.patch_proj = nn.Linear(patch_len, d_model)
        
        # 3. Positional Encoding (학습 가능한 파라미터 사용)
        self.W_pos = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        
        # 4. Transformer Encoder (Shared Encoder의 핵심)
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
        # 채널 수(M) * 패치 수 * d_model 을 Flatten 한 뒤 z_dim으로 압축
        self.flatten_dim = input_dim * self.num_patches * d_model
        self.fc = nn.Linear(self.flatten_dim, z_dim)

    def forward(self, x):
        """
        x shape: (Batch_size, Seq_len, Input_dim) -> (B, L, M)
        """
        B, L, M = x.shape
        
        # [Step 1] Channel Independence (채널 독립성)
        # 피처(채널)를 배치 차원으로 합쳐서 각 채널이 독립적인 시계열인 것처럼 처리
        # (B, L, M) -> (B, M, L) -> (B * M, L)
        x = x.permute(0, 2, 1).reshape(B * M, L)
        
        # [Step 2] Patching (패치화)
        # unfold를 사용하여 시퀀스를 패치 단위로 분할
        # (B * M, L) -> (B * M, num_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # [Step 3] Patch Embedding & Positional Encoding
        # (B * M, num_patches, patch_len) -> (B * M, num_patches, d_model)
        x = self.patch_proj(x) 
        x = x + self.W_pos
        
        # [Step 4] Transformer Encoder (모든 채널이 동일한 가중치 공유)
        # (B * M, num_patches, d_model)
        x = self.transformer_encoder(x)
        
        # [Step 5] 형태 복구 및 TNC용 최종 벡터 생성
        # (B * M, num_patches, d_model) -> (B, M, num_patches, d_model)
        x = x.view(B, M, self.num_patches, self.d_model)
        
        # 채널과 패치 정보를 모두 Flatten
        # (B, M * num_patches * d_model)
        x = x.reshape(B, -1)
        
        # (B, z_dim) - 최종적으로 Discriminator에 들어갈 단일 벡터 z 생성
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