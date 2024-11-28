import os
import glob
import re
import string
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 랜덤 시드 설정 (재현성 확보)
def set_seed(seed=42):
    
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

def evaluate(model, val_loader, thresholds):
    """
    모델을 평가하고 각 라벨별 정확도를 계산하여 평균 정확도, F1 점수, 그리고 손실을 반환합니다.
    """
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0  # 검증 손실 합계를 저장할 변수

    loss_fct = nn.BCEWithLogitsLoss()

    thresholds = np.array(thresholds)  # numpy 배열로 변환

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
            preds = (probs > thresholds).astype(int)
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    # 평균 손실 계산
    avg_loss = total_loss / len(val_loader)

    # numpy 배열로 변환
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # 각 라벨별 정확도 계산
    accuracies = []
    dichotomies = ['E/I', 'S/N', 'T/F', 'J/P']
    for i in range(true_labels.shape[1]):
        acc = accuracy_score(true_labels[:, i], predictions[:, i])
        accuracies.append(acc)
        print(f"{dichotomies[i]} 정확도: {acc * 100:.2f}%")

    # 평균 정확도 계산
    avg_accuracy = np.mean(accuracies)
    print(f"평균 정확도: {avg_accuracy * 100:.2f}%")

    # F1 Score 계산 (macro 평균)
    f1 = f1_score(true_labels, predictions, average='macro')

    return avg_accuracy, f1, avg_loss  # 평균 손실 반환

# ---------------------------
# Additional Functions
# ---------------------------

def predict(text, model, tokenizer, thresholds):
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
    
    thresholds = np.array(thresholds)  # numpy 배열로 변환

    with torch.no_grad():
        logits = model(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            token_type_ids=inputs['token_type_ids']
        )
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        predictions = (probs > thresholds).astype(int)
    
    # 예측된 dichotomy 변환
    e_i = 'E' if predictions[0] == 1 else 'I'
    s_n = 'S' if predictions[1] == 1 else 'N'
    t_f = 'T' if predictions[2] == 1 else 'F'
    j_p = 'J' if predictions[3] == 1 else 'P'
    
    predicted_mbti = e_i + s_n + t_f + j_p
    return predicted_mbti

def optimize_thresholds(model, val_loader):
    """
    검증 세트를 활용하여 각 라벨별 최적의 임계값을 찾습니다.
    """

    model.eval()
    true_labels = []
    logits_list = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].cpu().numpy().astype(int)  # 정수형으로 변환
            true_labels.extend(labels)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            logits_list.extend(logits.cpu().numpy())

    true_labels = np.array(true_labels, dtype=int)  # 정수형으로 변환
    logits = np.array(logits_list)
    probs = sigmoid(logits)

    thresholds = []
    for i in range(true_labels.shape[1]):
        best_threshold = 0.5
        best_accuracy = 0.0
        for threshold in np.linspace(0, 1, 101):
            preds = (probs[:, i] >= threshold).astype(int)
            accuracy = accuracy_score(true_labels[:, i], preds)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        thresholds.append(best_threshold)
        print(f"라벨 {i}의 최적 임계값: {best_threshold:.2f}, Accuracy: {best_accuracy:.4f}")
    return thresholds

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_model(model_class, num_labels=4):
    """
    저장된 ELECTRA와 Classifier를 로드하여 모델을 반환합니다.
    """
    model = model_class(num_labels=num_labels)
    model.electra.load_state_dict(torch.load('./models/model3/model3_electra_state.pth', map_location=device), strict=False)
    model.classifier.load_state_dict(torch.load('./models/model3/model3_classifier_state.pth', map_location=device), strict=False)
    model.to(device)
    model.eval()
    print("모델 로드 완료: model3_electra_state.pth, model3_classifier_state.pth")
    return model

def load_ablation_model(model_class, num_labels=4):
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
# Main Execution
# ---------------------------

if __name__ == "__main__":
    # 설정
    CSV_DIRECTORY = './data/comparison'  # 실제 CSV 파일들이 저장된 디렉토리 경로로 변경할 것
    EPOCHS = 20
    BATCH_SIZE = 16
    PATIENCE = 4
    FREEZE_LAYERS = 0  # 동결할 ELECTRA 층의 수

    # 각 라벨별 임계값 설정 (E/I, S/N, T/F, J/P 순서)
    thresholds = [0.5, 0.5, 0.5, 0.5]

    # 1. Data Preparation
    df_new = prepare_data(CSV_DIRECTORY)

    # 2. Model Architecture
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
    
    # 데이터 분할: 훈련(70%), 검증(15%), 테스트(15%)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df_new['Article'],
        df_new[['E_I', 'S_N', 'T_F', 'J_P']],
        test_size=0.9,
        random_state=42,
        stratify=df_new[['E_I', 'S_N', 'T_F', 'J_P']]
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels
    )


    # 데이터셋 생성
    train_dataset = MBTIDataset(train_texts.tolist(), train_labels.values.tolist(), tokenizer)
    val_dataset = MBTIDataset(val_texts.tolist(), val_labels.values.tolist(), tokenizer)
    test_dataset = MBTIDataset(test_texts.tolist(), test_labels.values.tolist(), tokenizer)
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 모델 로드
    our_model = load_model(ElectraForCustomClassification)

    # 검증 세트를 활용하여 최적의 임계값 찾기
    print("\n--- 검증 세트를 활용한 임계값 최적화 ---")
    thresholds = optimize_thresholds(our_model, val_loader)
    print(f"최적화된 임계값: {thresholds}")
 
    # 6. Test Set Evaluation for each model
    print("\n--- Our Model 테스트 세트 평가 ---")
    test_accuracy, test_f1, test_loss = evaluate(our_model, test_loader, thresholds)
    print(f"Our Model 테스트 손실: {test_loss:.4f}")
    print(f"Our Model 테스트 정확도: {test_accuracy * 100:.2f}%")
    print(f"Our Model 테스트 F1 점수: {test_f1 * 100:.2f}%")

    ablation_model = load_ablation_model(ElectraForCustomClassification)
    ablation_test_accuracy, ablation_test_f1, ablation_test_loss = evaluate(ablation_model, test_loader, [0.5, 0.5, 0.5, 0.5])
    print(f"Our Model 테스트 손실: {ablation_test_loss:.4f}")
    print(f"Our Model 테스트 정확도: {ablation_test_accuracy * 100:.2f}%")
    print(f"Our Model 테스트 F1 점수: {ablation_test_f1 * 100:.2f}%")
        
    # 8. 예측 예시
    example_text = "너가 기존에 masking한 방식은 gpt한테 부탁한거라 그랬지??! 오케이오케이 그런 식으로 갈 것 같어 고마워!"
    predicted_mbti1 = predict(example_text, our_model, tokenizer, thresholds)
    print(f"\nOur Model이 예측한 MBTI 유형: {predicted_mbti1}")
    predicted_mbti2 = predict(example_text, ablation_model, tokenizer, thresholds)
    print(f"\nAblation Study Model이 예측한 MBTI 유형: {predicted_mbti2}")
