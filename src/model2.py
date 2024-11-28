import os
import random
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import math  # Perplexity 계산을 위해 추가

# 랜덤 시드 설정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 중인 장치: {device}')

# MBTI 유형 및 매핑
MBTI_TYPES = [
    'INTJ', 'INTP', 'ENTJ', 'ENTP',
    'INFJ', 'INFP', 'ENFJ', 'ENFP',
    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
    'ISTP', 'ISFP', 'ESTP', 'ESFP'
]
mbti_to_id = {mbti: idx for idx, mbti in enumerate(MBTI_TYPES)}

# ---------------------------
# 1. Data Preparation
# ---------------------------

# 텍스트 정제 함수
def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[\r\n]', ' ', text)
    text = re.sub(r'[^가-힣a-zA-Z0-9\s\[\]{}()]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------
# 2. Model Architecture
# ---------------------------

# 데이터셋 클래스
class MBTIMaskedDataset(Dataset):
    def __init__(self, csv_file, tokenizer, mbti_to_id, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mbti_to_id = mbti_to_id

        # 문장 정제
        self.data['masked_sentence'] = self.data['masked_sentence'].apply(clean_text)
        self.data['original_sentence'] = self.data['original_sentence'].apply(clean_text)

        # [CLS] 토큰을 마스킹하지 않도록 확인
        # 만약 [CLS] 토큰이 마스킹되었다면, 이를 원본으로 복원
        # 이는 데이터 전처리 단계에서 [CLS] 토큰이 마스킹되지 않았다는 것을 보장합니다.
        self.data['masked_sentence'] = self.data.apply(
            lambda row: row['masked_sentence'].replace(tokenizer.cls_token, tokenizer.cls_token) if tokenizer.cls_token in row['masked_sentence'] else row['masked_sentence'],
            axis=1
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        masked_sentence = self.data.iloc[idx]['masked_sentence']
        original_sentence = self.data.iloc[idx]['original_sentence']
        mbti = self.data.iloc[idx]['MBTI']
        mbti_idx = self.mbti_to_id[mbti]

        # 마스킹된 문장 토크나이즈
        inputs = self.tokenizer(
            masked_sentence,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=True
        )

        # 레이블 생성
        labels = self.tokenizer(
            original_sentence,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=True
        )['input_ids']

        # 텐서 형태 변환
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = labels.squeeze(0)

        # [CLS], [SEP] 토큰을 레이블에서 무시
        ignore_tokens = [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]
        for token_id in ignore_tokens:
            labels[labels == token_id] = -100

        mbti_idx = torch.tensor(mbti_idx)

        return inputs['input_ids'], inputs['attention_mask'], labels, mbti_idx
    

# 모델 클래스
class ElectraForMaskedLMWithMBTI(nn.Module):
    def __init__(self, electra_model_name, num_mbti_types, mbti_embedding_dim=128):
        super(ElectraForMaskedLMWithMBTI, self).__init__()
        self.electra = AutoModel.from_pretrained(electra_model_name)
        self.hidden_size = self.electra.config.hidden_size

        # MBTI 임베딩
        self.mbti_embedding = nn.Embedding(num_mbti_types, mbti_embedding_dim)

        # 결합을 위한 선형층
        self.fc = nn.Linear(self.hidden_size + mbti_embedding_dim, self.electra.config.vocab_size)

    def forward(self, input_ids, attention_mask, mbti_idx):
        # Electra 출력
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # MBTI 임베딩
        mbti_embed = self.mbti_embedding(mbti_idx)  # (batch_size, mbti_embedding_dim)
        mbti_embed = mbti_embed.unsqueeze(1).expand(-1, sequence_output.size(1), -1)  # (batch_size, seq_len, mbti_embedding_dim)

        # 결합
        combined = torch.cat((sequence_output, mbti_embed), dim=-1)  # (batch_size, seq_len, hidden_size + mbti_embedding_dim)

        # 로짓 계산
        logits = self.fc(combined)  # (batch_size, seq_len, vocab_size)

        return logits
    
# ---------------------------
# 3. Train, Validate, Test and Experiment
# ---------------------------

# 학습 함수
def train(model, dataloader, optimizer, scheduler, criterion, tokenizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids, attention_mask, labels, mbti_idx = [b.to(device) for b in batch]

        # 패딩 토큰 및 특별 토큰 무시
        # 이미 데이터셋 클래스에서 [CLS], [SEP], [PAD] 토큰을 -100으로 설정했으므로 추가 설정 필요 없음

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, mbti_idx)
        logits = outputs

        # [CLS] 토큰의 로짓을 매우 낮은 값으로 설정하여 예측 후보에서 제외
        cls_token_id = tokenizer.cls_token_id
        if cls_token_id is not None:
            logits[:, :, cls_token_id] = -1e9  # 또는 -float('inf')

        # [SEP] 토큰도 예측 후보에서 제외
        sep_token_id = tokenizer.sep_token_id
        if sep_token_id is not None:
            logits[:, :, sep_token_id] = -1e9  # 또는 -float('inf')

        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# 평가 함수
def evaluate(model, dataloader, criterion, tokenizer, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels, mbti_idx = [b.to(device) for b in batch]

            # 패딩 토큰 및 특별 토큰 무시
            # 이미 데이터셋 클래스에서 [CLS], [SEP], [PAD] 토큰을 -100으로 설정했으므로 추가 설정 필요 없음

            outputs = model(input_ids, attention_mask, mbti_idx)
            logits = outputs

            # [CLS] 토큰의 로짓을 매우 낮은 값으로 설정하여 예측 후보에서 제외
            cls_token_id = tokenizer.cls_token_id
            if cls_token_id is not None:
                logits[:, :, cls_token_id] = -1e9  # 또는 -float('inf')

            # [SEP] 토큰도 예측 후보에서 제외
            sep_token_id = tokenizer.sep_token_id
            if sep_token_id is not None:
                logits[:, :, sep_token_id] = -1e9  # 또는 -float('inf')

            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)  # Perplexity 계산

    return avg_loss, perplexity
# ---------------------------
# 4. Visualization of Result
# ---------------------------

def plot_training_history(history, test_loss=None, test_perplexity=None, save_path='training_history.png'):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 6))  # 크기 조정

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Per Epoch')
    plt.legend()

    # Perplexity 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_perplexity'], label='Validation Perplexity', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity Per Epoch')
    plt.legend()

    # 테스트 결과 추가
    if test_loss is not None and test_perplexity is not None:
        plt.figtext(0.5, 0.01, f'Test Loss: {test_loss:.4f} | Test Perplexity: {test_perplexity:.2f}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# ---------------------------
# 5. Manage (Save and Load) Model
# ---------------------------

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"모델 저장: {path}")

def load_model(model_class, model_path, electra_model_name, num_mbti_types, device):
    model = model_class(electra_model_name, num_mbti_types)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    print(f"모델 로드: {model_path}")
    return model

def save_electra(model, path):
    torch.save(model.electra.state_dict(), path)
    print(f"Electra 모델 저장: {path}")

def load_electra(model, path, device):
    model.electra.load_state_dict(torch.load(path, map_location=device), strict=False)
    model.to(device)
    print(f"Electra 모델 로드: {path}")


# ---------------------------
# Additional Functions
# ---------------------------

def predict_masked_word(model, tokenizer, sentence, mbti_type, mbti_to_id, device, top_k=5):
    """
    마스크된 문장에서 가장 중요한 단어를 예측합니다.

    Args:
        model: 학습된 모델.
        tokenizer: 토크나이저.
        sentence: 마스크된 문장.
        mbti_type: 해당 문장의 MBTI 유형.
        mbti_to_id: MBTI 유형을 인덱스로 매핑한 딕셔너리.
        device: 학습 장치.
        top_k: 예측할 상위 k개의 단어.

    Returns:
        예측된 단어와 그 확률의 리스트.
    """
    model.eval()
    
    if mbti_type not in mbti_to_id:
        print(f"MBTI type '{mbti_type}' not recognized.")
        return None
    
    mbti_idx = torch.tensor(mbti_to_id[mbti_type]).unsqueeze(0).to(device)
    
    # 문장 토크나이즈
    inputs = tokenizer(
        sentence,
        return_tensors='pt',
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], mbti_idx=mbti_idx)
    
    # [MASK] 토큰의 위치 찾기
    mask_token_indices = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)
    if len(mask_token_indices[0]) == 0:
        print("No [MASK] token found in the sentence.")
        return None
    
    # [MASK] 토큰의 로짓 추출
    mask_logits = outputs[mask_token_indices[0], mask_token_indices[1], :]
    
    # [CLS] 토큰의 로짓을 매우 낮은 값으로 설정하여 예측 후보에서 제외
    cls_token_id = tokenizer.cls_token_id
    if cls_token_id is not None:
        mask_logits[:, cls_token_id] = -1e9  # 또는 -float('inf')
    
    # [SEP] 토큰도 예측 후보에서 제외
    sep_token_id = tokenizer.sep_token_id
    if sep_token_id is not None:
        mask_logits[:, sep_token_id] = -1e9  # 또는 -float('inf')
    
    # Softmax를 적용하여 확률 분포 생성
    mask_probs = torch.softmax(mask_logits, dim=1)
    
    # Top-K 예측 단어 및 확률 추출
    top_probs, top_indices = mask_probs.topk(top_k, dim=1)
    
    predictions = []
    for i in range(top_k):
        token = tokenizer.decode(top_indices[0, i]).strip()
        prob = top_probs[0, i].item()
        predictions.append((token, prob))
    
    return predictions

# ---------------------------
# Main Execution
# ---------------------------

if __name__ == '__main__':
    # 설정
    electra_model_name = "beomi/KcELECTRA-base-v2022"
    batch_size = 16
    num_epochs = 20
    learning_rate = 5e-5
    max_length = 256
    csv_file = './data/model2/masked_data.csv'  # 실제 CSV 파일 경로로 변경
    patience = 3  # 조기 종료 기준

    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(electra_model_name)

    # 데이터셋 초기화
    dataset = MBTIMaskedDataset(csv_file, tokenizer, mbti_to_id, max_length=max_length)

    # 데이터셋 분할 (stratify 인자 제거)
    indices = list(range(len(dataset)))
    train_indices, val_test_indices = train_test_split(
        indices,
        test_size=0.3,
        random_state=42
    )
    val_indices, test_indices = train_test_split(
        val_test_indices,
        test_size=0.5,
        random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_mbti_types = len(mbti_to_id)

    # 모델 초기화
    model = ElectraForMaskedLMWithMBTI(electra_model_name, num_mbti_types).to(device)

    # Electra 가중치 로드
    electra_weights_path = './models/model1/model1_electra_state.pth'
    if os.path.exists(electra_weights_path):
        load_electra(model, electra_weights_path, device)
    else:
        print("Electra 가중치를 찾을 수 없어 사전 학습된 모델을 사용합니다.")

    # 옵티마이저, 스케줄러, 손실 함수 정의
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 학습 루프
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_perplexity': []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_dataloader, optimizer, scheduler, criterion, tokenizer, device)
        val_loss, val_perplexity = evaluate(model, val_dataloader, criterion, tokenizer, device)

        # 기록 저장
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_perplexity'].append(val_perplexity)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Perplexity: {val_perplexity:.2f}")

        # 성능 향상 확인
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_model(model, './models/model2/model2_model_state.pth')
            save_electra(model, './models/model2/model2_electra_state.pth')
            print("최고 성능 모델 저장됨.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("조기 종료: 검증 손실 개선 없음.")
                break

    # 최고의 모델 로드
    model = load_model(ElectraForMaskedLMWithMBTI, './models/model2/model2_model_state.pth', electra_model_name, num_mbti_types, device)

    # 테스트 세트 평가
    test_loss, test_perplexity = evaluate(model, test_dataloader, criterion, tokenizer, device)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {test_perplexity:.2f}")

    # 학습 기록 시각화 및 저장
    plot_training_history(history, test_loss=test_loss, test_perplexity=test_perplexity, save_path='training_history.png')

    # 예측 예시 (필요한 경우 사용)
    example_sentence = "내일 모임 시간 딱 지켜야 해. 늦으면 곤란하니까. 할 일은 미리미리 [MASK]야 마음이 편하지 않겠어?"
    example_mbti = 'ISTJ'  # 원하는 MBTI 유형으로 변경

    predictions = predict_masked_word(model, tokenizer, example_sentence, example_mbti, mbti_to_id, device)
    if predictions:
        print("\nPredictions:")
        for idx, (token, prob) in enumerate(predictions):
            print(f"Option {idx+1}: {token} (Probability: {prob:.4f})")
