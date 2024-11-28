import os
import glob
import re
import string
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

# 랜덤 시드 설정 (재현성 확보)
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 중인 장치: {device}')

# ---------------------------
# 1. Data Preparation
# ---------------------------


# MBTI 유형을 4개의 dichotomy로 분할하여 이진 레이블로 변환
def split_mbti(mbti):
    mbti = mbti.upper()
    e_i = 1 if mbti[0] == 'E' else 0
    s_n = 1 if mbti[1] == 'S' else 0
    t_f = 1 if mbti[2] == 'T' else 0
    j_p = 1 if mbti[3] == 'J' else 0
    return [e_i, s_n, t_f, j_p]

def replace_mbti_types_with_person(text, mbti_list, replacement='사람'):
    """
    텍스트 내의 MBTI 유형을 "사람"으로 대체합니다.
    """
    pattern = re.compile(r'\b(' + '|'.join(mbti_list) + r')\b', flags=re.IGNORECASE)
    cleaned_text = pattern.sub(replacement, text)
    return cleaned_text

def clean_text(text):
    """
    텍스트를 정제하여 불필요한 문자와 공백을 제거합니다.
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_two_sentences(text):
    """
    텍스트를 2문장씩 분할하여 리스트로 반환합니다.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    new_texts = []
    for i in range(0, len(sentences), 2):
        new_text = ' '.join(sentences[i:i+2])
        new_texts.append(new_text)
    return new_texts

def load_multiple_csv(csv_directory, csv_extension="*.csv"):
    """
    지정된 디렉토리에서 모든 CSV 파일을 불러와 하나의 데이터프레임으로 병합합니다.
    """
    csv_files = glob.glob(os.path.join(csv_directory, csv_extension))
    if not csv_files:
        raise ValueError(f"지정된 디렉토리 '{csv_directory}'에 '{csv_extension}' 패턴에 맞는 파일이 없습니다.")
    
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
            print(f"'{file}' 파일을 성공적으로 불러왔습니다. (행 수: {len(df)})")
        except Exception as e:
            print(f"'{file}' 파일을 불러오는 중 오류 발생: {e}")
    
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"\n모든 CSV 파일을 병합한 데이터프레임의 총 행 수: {len(merged_df)}")
    return merged_df

def prepare_data(csv_directory):
    """
    전체 데이터 준비 과정을 수행합니다.
    """
    # MBTI 유형 리스트
    MBTI_TYPES = [
        'ISTJ', 'ISFJ', 'INFJ', 'INTJ',
        'ISTP', 'ISFP', 'INFP', 'INTP',
        'ESTP', 'ESFP', 'ENFP', 'ENTP',
        'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ'
    ]

    # CSV 파일 로드
    df = load_multiple_csv(csv_directory)

    df.sample()    
    # 데이터 확인
    print("\n원본 데이터:")
    print(df.head())
    
    # 결측치 제거
    print("\n결측치 제거 전 데이터 크기:", df.shape)
    df = df.dropna()  # 결측치가 있는 행 제거
    print("결측치 제거 후 데이터 크기:", df.shape)
    
    # MBTI 유형 대체
    df['Article'] = df['Article'].astype(str).apply(lambda x: replace_mbti_types_with_person(x, MBTI_TYPES))
    
    # 추가 텍스트 정제
    df['Article'] = df['Article'].apply(lambda x: clean_text(x))
    
    # 텍스트 변환 후 샘플 출력
    print("\nMBTI 유형 대체 후 텍스트 샘플:")
    print(df[['MBTI', 'Article']].head())
    
    df[['E_I', 'S_N', 'T_F', 'J_P']] = df['MBTI'].apply(split_mbti).tolist()
    print("\n라벨 분할 후 데이터 샘플:")
    print(df[['MBTI', 'E_I', 'S_N', 'T_F', 'J_P']].head())
    
    # 문장 분할
    new_data = {'Article': [], 'E_I': [], 'S_N': [], 'T_F': [], 'J_P': []}
    for idx, row in df.iterrows():
        articles = split_into_two_sentences(row['Article'])
        e_i, s_n, t_f, j_p = row['E_I'], row['S_N'], row['T_F'], row['J_P']
        for article in articles:
            article = article.strip()
            if article:
                new_data['Article'].append(article)
                new_data['E_I'].append(e_i)
                new_data['S_N'].append(s_n)
                new_data['T_F'].append(t_f)
                new_data['J_P'].append(j_p)
    
    df_new = pd.DataFrame(new_data)
    print("\n분할된 데이터:")
    print(df_new.head())
    
    return df_new


# ---------------------------
# 2. Model Architecture
# ---------------------------

# ---------------------------
# 2. Model Architecture
# ---------------------------

class ElectraForCustomClassification(nn.Module):
    def __init__(self, num_labels=4, dropout_rate=0.3, freeze_layers=0):
        super(ElectraForCustomClassification, self).__init__()
        self.electra = AutoModel.from_pretrained("beomi/KcELECTRA-base-v2022")
        
        # 특정 층 동결 (옵션)
        if freeze_layers > 0:
            for layer in self.electra.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # MLP Classifier 정의
        self.classifier = nn.Sequential(
            nn.Linear(self.electra.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_labels)
        )
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.electra(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰의 출력
        
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)
        
        return logits

# ---------------------------
# 3. Train, Validate, Test and Experiment
# ---------------------------

class MBTIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels  # labels는 [E_I, S_N, T_F, J_P]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),  # (max_length,)
            'attention_mask': encoding['attention_mask'].flatten(),  # (max_length,)
            'token_type_ids': encoding['token_type_ids'].flatten(),  # (max_length,)
            'labels': torch.tensor(label, dtype=torch.float)  # (4,)
        }

def train_validate(model, train_loader, val_loader, optimizer, scheduler, epochs, patience=3):
    """
    모델을 학습하고 검증하는 함수입니다.
    """
    best_f1 = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}  # val_loss 추가

    for epoch in range(epochs):
        print(f'\n===== 에포크 {epoch + 1}/{epochs} =====')
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc='훈련 진행 중')

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
            )
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'손실': f'{loss.item():.4f}'})

        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f'평균 훈련 손실: {avg_train_loss:.4f}')

        # 평가
        val_accuracy, val_f1, val_loss = evaluate(model, val_loader)  # val_loss 추가
        history['val_accuracy'].append(val_accuracy)
        history['val_f1'].append(val_f1)
        history['val_loss'].append(val_loss)  # val_loss 저장
        print(f'검증 손실: {val_loss:.4f}')  # val_loss 출력
        print(f'검증 정확도: {val_accuracy * 100:.2f}%')
        print(f'검증 F1 점수: {val_f1 * 100:.2f}%')

        # 조기 종료 조건 확인
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            # 모델 저장 (ELECTRA와 Classifier를 각각 저장)
            save_model(model)
            print(f'최고 성능 모델 저장: ablation_model3_electra_state.pth, ablation_model3_classifier_state.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("조기 종료 조건 만족. 학습을 중단합니다.")
                break

    return history

def evaluate(model, val_loader):
    """
    모델을 평가하고 정확도, F1 점수, 그리고 손실을 반환합니다.
    """
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0  # 검증 손실 합계를 저장할 변수

    loss_fct = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
            )

            # 손실 계산
            loss = loss_fct(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    # 평균 손실 계산
    avg_loss = total_loss / len(val_loader)

    # F1 Score 계산 (macro 평균)
    f1 = f1_score(true_labels, predictions, average='macro')
    # Accuracy Score 계산 (Subset Accuracy)
    acc = accuracy_score(true_labels, predictions)
    return acc, f1, avg_loss  # 평균 손실 반환

# ---------------------------
# 4. Visualization of Result
# ---------------------------

def plot_history(history):
    """
    학습 과정의 손실 및 검증 지표를 시각화합니다.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    sns.lineplot(x=epochs, y=history['train_loss'], label='Train Loss')
    sns.lineplot(x=epochs, y=history['val_loss'], label='Validation Loss')  # 검증 손실 추가
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()

    # F1 점수 그래프
    plt.subplot(1, 2, 2)
    sns.lineplot(x=epochs, y=history['val_accuracy'], label='Validation Accuracy')
    sns.lineplot(x=epochs, y=history['val_f1'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(model, val_loader, threshold=0.5):
    """
    각 dichotomy별 혼동 행렬을 시각화합니다.
    """
    from sklearn.metrics import confusion_matrix
    
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)
            all_preds.append(preds)
            all_true.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_true = np.vstack(all_true)
    
    dichotomies = ['E/I', 'S/N', 'T/F', 'J/P']
    
    for i, dichotomy in enumerate(dichotomies):
        cm = confusion_matrix(all_true[:, i], all_preds[:, i])
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {dichotomy}')
        
        # 슬래시('/')를 하이픈('-')으로 대체하여 유효한 파일 이름 생성
        safe_dichotomy = dichotomy.replace('/', '-')
        plt.savefig(f'confusion_matrix_{safe_dichotomy}.png')
        plt.show()

def plot_mbti_type_accuracy(model, val_loader, threshold=0.5):
    """
    16가지 MBTI 유형(ISFJ, ISTJ 등)의 분류 정확도를 계산하고 시각화합니다.
    """
    MBTI_TYPES = [
        'ISTJ', 'ISFJ', 'INFJ', 'INTJ',
        'ISTP', 'ISFP', 'INFP', 'INTP',
        'ESTP', 'ESFP', 'ENFP', 'ENTP',
        'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ'
    ]
    
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)
            
            all_preds.extend(preds)
            all_true.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    # MBTI 유형으로 변환
    def get_mbti_label(row):
        return ''.join([
            'E' if row[0] == 1 else 'I',
            'S' if row[1] == 1 else 'N',
            'T' if row[2] == 1 else 'F',
            'J' if row[3] == 1 else 'P'
        ])
    
    predicted_mbti = [get_mbti_label(row) for row in all_preds]
    true_mbti = [get_mbti_label(row) for row in all_true]
    
    # 정확도 계산
    accuracy_by_type = {}
    for mbti in MBTI_TYPES:
        true_count = sum([1 for t, p in zip(true_mbti, predicted_mbti) if t == mbti and p == mbti])
        total_count = sum([1 for t in true_mbti if t == mbti])
        accuracy_by_type[mbti] = true_count / total_count if total_count > 0 else 0.0
    
    # 정확도 시각화
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=list(accuracy_by_type.keys()), 
        y=list(accuracy_by_type.values())
    )
    plt.ylim(0, 1)
    plt.title('MBTI Type Classification Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('MBTI Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('mbti_type_accuracy.png')
    plt.show()
    
    print("MBTI 유형별 정확도 저장 완료: mbti_type_accuracy.png")

# ---------------------------
# 5. Manage (Save and Load) Experiment Result and Model
# ---------------------------

def save_experiment_history(history):
    """
    학습 이력을 저장합니다.
    """
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print(f"학습 이력 저장 완료: training_history.pkl")

def load_experiment_history():
    """
    저장된 학습 이력을 불러옵니다.
    """
    with open('training_history.pkl', 'rb') as f:
        history = pickle.load(f)
    print(f"학습 이력 불러오기 완료: training_history.pkl")
    return history

def save_model(model):
    """
    모델의 ELECTRA 부분과 Classifier 부분을 각각 저장합니다.
    """
    torch.save(model.electra.state_dict(), './models/ablation_study_model/ablation_model3_electra_state.pth')
    torch.save(model.classifier.state_dict(), './models/ablation_study_model/ablation_model3_classifier_state.pth')

def load_model(model_class, num_labels=4):
    """
    저장된 ELECTRA와 Classifier를 로드하여 모델을 반환합니다.
    """
    model = model_class(num_labels=num_labels)
    model.electra.load_state_dict(torch.load('./models/ablation_study_model/ablation_model3_electra_state.pth', map_location=device), strict=False)
    model.classifier.load_state_dict(torch.load('./models/ablation_study_model/ablation_model3_classifier_state.pth', map_location=device), strict=False)
    model.to(device)
    model.eval()
    print("모델 로드 완료: ablation_model3_electra_state.pth, ablation_model3_classifier_state.pth")
    return model

# ---------------------------
# Additional Functions
# ---------------------------

def predict(text, model, tokenizer, threshold=0.5):
    """
    입력된 텍스트에 대해 MBTI 유형을 예측합니다.
    """
    model.eval()
    MBTI_TYPES = [
        'ISTJ', 'ISFJ', 'INFJ', 'INTJ',
        'ISTP', 'ISFP', 'INFP', 'INTP',
        'ESTP', 'ESFP', 'ENFP', 'ENTP',
        'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ'
    ]
    text = replace_mbti_types_with_person(text, mbti_list=MBTI_TYPES)
    text = clean_text(text)
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    inputs = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        logits = model(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            token_type_ids=inputs['token_type_ids']
        )
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        predictions = (probs > threshold).astype(int)
    
    # 예측된 dichotomy 변환
    e_i = 'E' if predictions[0] == 1 else 'I'
    s_n = 'S' if predictions[1] == 1 else 'N'
    t_f = 'T' if predictions[2] == 1 else 'F'
    j_p = 'J' if predictions[3] == 1 else 'P'
    
    predicted_mbti = e_i + s_n + t_f + j_p
    return predicted_mbti

# ---------------------------
# Main Execution
# ---------------------------

if __name__ == "__main__":
    # 설정
    CSV_DIRECTORY = './data/ablation_study_model/'  # 실제 CSV 파일들이 저장된 디렉토리 경로로 변경할 것
    EPOCHS = 20
    BATCH_SIZE = 16
    PATIENCE = 4
    FREEZE_LAYERS = 0  # 동결할 ELECTRA 층의 수
    
    # 1. Data Preparation
    df_new = prepare_data(CSV_DIRECTORY)

    # 2. Model Architecture
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
    
    # 데이터 분할: 훈련(70%), 검증(15%), 테스트(15%)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df_new['Article'],
        df_new[['E_I', 'S_N', 'T_F', 'J_P']],
        test_size=0.3,
        random_state=42
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        random_state=42
    )

    # 데이터셋 생성
    train_dataset = MBTIDataset(train_texts.tolist(), train_labels.values.tolist(), tokenizer)
    val_dataset = MBTIDataset(val_texts.tolist(), val_labels.values.tolist(), tokenizer)
    test_dataset = MBTIDataset(test_texts.tolist(), test_labels.values.tolist(), tokenizer)
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 초기화
    model = ElectraForCustomClassification(num_labels=4, dropout_rate=0.3, freeze_layers=FREEZE_LAYERS)
    model.to(device)
    
    # 옵티마이저와 스케줄러 정의
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 3. Train, Validate, Test and Experiment
    history = train_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        patience=PATIENCE
    )
    
    # 모델 저장은 학습 함수 내에서 자동으로 이루어집니다.
    
    # 4. Visualization of Result
    plot_history(history)
    plot_confusion_matrix(model, val_loader, threshold=0.5)
    plot_mbti_type_accuracy(model, val_loader, threshold=0.5)
    
    # 5. Manage (Save and Load) Experiment Result and Model
    save_experiment_history(history)
    
    # 6. Test Set Evaluation
    print("\n--- 테스트 세트 평가 ---")
    test_accuracy, test_f1, test_loss = evaluate(model, test_loader)
    print(f"테스트 손실: {test_loss:.4f}")
    print(f"테스트 정확도: {test_accuracy * 100:.2f}%")
    print(f"테스트 F1 점수: {test_f1 * 100:.2f}%")
    
    # 테스트 세트에 대한 상세 분류 보고서 출력
    print("\n--- 테스트 세트 상세 분류 보고서 ---")
    model.eval()
    test_predictions = []
    test_true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            test_predictions.extend(preds)
            test_true_labels.extend(labels.cpu().numpy())
    
    dichotomies = ['E/I', 'S/N', 'T/F', 'J/P']
    for i, dichotomy in enumerate(dichotomies):
        print(f"\n--- {dichotomy} ---")
        print(classification_report(
            [label[i] for label in test_true_labels],
            [pred[i] for pred in test_predictions],
            target_names=['Negative', 'Positive']
        ))
    
    # 7. 모델 로드 테스트
    print("\n--- 모델 로드 테스트 ---")
    loaded_model = load_model(ElectraForCustomClassification)
    # 로드한 모델로 테스트 세트 평가
    loaded_test_accuracy, loaded_test_f1, loaded_test_loss = evaluate(loaded_model, test_loader)
    print(f"로드된 모델 테스트 손실: {loaded_test_loss:.4f}")
    print(f"로드된 모델 테스트 정확도: {loaded_test_accuracy * 100:.2f}%")
    print(f"로드된 모델 테스트 F1 점수: {loaded_test_f1 * 100:.2f}%")
        
    # 8. 예측 예시 (ENTP)
    example_text = "나는 지적인 도전을 즐기며, 문제 해결에 큰 흥미를 느낍니다."
    predicted_mbti = predict(example_text, loaded_model, tokenizer)
    print(f"\n예측된 MBTI 유형: {predicted_mbti}")
