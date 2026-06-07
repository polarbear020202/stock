import config
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAUROC, MulticlassAveragePrecision
from tqdm import tqdm
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TFC(nn.Module):
    def __init__(self, ts_length, num_features, num_layers, nhead, feat_dim):
        super(TFC, self).__init__()

        encoder_layers_t = TransformerEncoderLayer(
            d_model=num_features,
            dim_feedforward=2 * num_features,
            nhead=nhead,
            batch_first=True
        )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, num_layers)

        flatten_dim = ts_length * num_features

        self.projector_t = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, feat_dim)
        )

        encoder_layers_f = TransformerEncoderLayer(
            d_model=num_features,
            dim_feedforward=2 * num_features,
            nhead=nhead,
            batch_first=True
        )
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, num_layers)

        self.projector_f = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, feat_dim)
        )

    def forward(self, x_in_t, x_in_f):
        x = self.transformer_encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)
        z_time = self.projector_t(h_time)

        f = self.transformer_encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq

class NTXentLoss(torch.nn.Module):
    def __init__(self, device, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, actual_batch_size):
        mask = torch.ones((2 * actual_batch_size, 2 * actual_batch_size), dtype=torch.bool, device=self.device)
        mask.fill_diagonal_(False)
        for i in range(actual_batch_size):
            mask[i, actual_batch_size + i] = False
            mask[actual_batch_size + i, i] = False
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_simililarity(self, x, y):
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        actual_batch_size = zis.shape[0]

        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)

        mask = self._get_correlated_mask(actual_batch_size)

        l_pos = torch.diag(similarity_matrix, actual_batch_size)
        r_pos = torch.diag(similarity_matrix, -actual_batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * actual_batch_size, 1)

        negatives = similarity_matrix[mask].view(2 * actual_batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * actual_batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * actual_batch_size)


class TFCTrainer:
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

        for x_t in progress_bar:
            x_t = x_t.to(self.device)

            x_f = torch.abs(torch.fft.fft(x_t))

            self.optimizer.zero_grad()

            _, z_time, _, z_freq = self.model(x_t, x_f)

            loss = self.criterion(z_time, z_freq)
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
            for x_t in progress_bar:
                x_t = x_t.to(self.device)

                x_f = torch.abs(torch.fft.fft(x_t))

                _, z_time, _, z_freq = self.model(x_t, x_f)

                loss = self.criterion(z_time, z_freq)

                total_loss += loss.item()
                progress_bar.set_postfix({'val_loss': f"{loss.item():.4f}"})

        return total_loss / len(self.val_loader)

    def fit(self):
        max_val_loss = float('inf')
        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch [{epoch}/{self.epochs}]")

            # 1. Train 호출
            train_loss = self.train_epoch()

            # 2. Validation 호출
            val_loss = self.validate_epoch()
            if val_loss < max_val_loss:
                max_val_loss = val_loss
                torch.save(self.model.state_dict(), 'TFC.pth')
                print("Model saved!")

            # 에폭 결과 출력
            print(f"-> Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

class target_classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        :param input_dim: TFC에서 나온 특징 벡터의 합산 차원 (2 * 128)
        :param num_classes: 분류할 클래스 개수 (configs.num_classes_target)
        """
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(input_dim, 64)
        self.logits_simple = nn.Linear(64, num_classes)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred


class HeadTrainer:
    def __init__(self, embed_model , model, train_loader, val_loader, optimizer, criterion, device, epochs, num_classes):
        self.embed_model = embed_model
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.num_classes = num_classes

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for x,y in progress_bar:
            x_t = x.to(self.device)
            y = y.long().to(self.device)
            x_f = torch.abs(torch.fft.fft(x_t))

            self.optimizer.zero_grad()
            _ , z_t , _,  z_f = self.embed_model(x_t, x_f)
            pred = self.model(torch.cat([z_t, z_f], dim=1))
            loss = self.criterion(pred, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        conf = {'num_classes': self.num_classes, 'average': 'macro'}.copy()
        metrics = {
            'acc': MulticlassAccuracy(num_classes=self.num_classes).to(self.device),
            'pre': MulticlassPrecision(**conf).to(self.device),
            'rec': MulticlassRecall(**conf).to(self.device),
            'f1': MulticlassF1Score(**conf).to(self.device),
            'auroc': MulticlassAUROC(num_classes=self.num_classes).to(self.device),
            'auprc': MulticlassAveragePrecision(num_classes=self.num_classes).to(self.device)
        }

        progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for x, y in progress_bar:
                x_t, y = x.to(self.device), y.long().to(self.device)
                x_f = torch.abs(torch.fft.fft(x_t))

                _ , z_t , _,  z_f = self.embed_model(x_t, x_f)
                pred = self.model(torch.cat([z_t, z_f], dim=1))

                loss = self.criterion(pred, y)
                total_loss += loss.item()

                for m in metrics.values(): m.update(pred, y)
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        res = {k: m.compute().item() for k, m in metrics.items()}
        res['loss'] = total_loss / len(self.val_loader)

        print(f"\n[Val] Loss: {res['loss']:.4f} | Acc: {res['acc']:.4f}| pre: {res['pre']:.4f} | rec: {res['rec']:.4f} | F1: {res['f1']:.4f} | AUROC: {res['auroc']:.4f}")
        return res['loss']

    def fit(self):
        max_val_loss = float('inf')
        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch [{epoch}/{self.epochs}]")
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            if val_loss < max_val_loss:
                max_val_loss = val_loss
                torch.save(self.model.state_dict(), 'TFC_head_model.pth')
                print("Model saved!")

            print(f"-> Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")