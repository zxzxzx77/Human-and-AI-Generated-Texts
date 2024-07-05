import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
df = pd.read_csv('/content/DatasetPart1-20-100Words.csv')
X = df['Text'].values
y = df['Class'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Load BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# Tokenize the input data
inputs = tokenizer(list(X), return_tensors='pt', padding=True, truncation=True, max_length=512)

# Split the indices for training and validation sets
train_indices, val_indices = train_test_split(range(len(y_encoded)), test_size=0.2, random_state=42)

# Create the DataLoader with the correct subsets
train_inputs = {key: val[train_indices] for key, val in inputs.items()}
val_inputs = {key: val[val_indices] for key, val in inputs.items()}

train_labels = torch.tensor(y_encoded[train_indices])
val_labels = torch.tensor(y_encoded[val_indices])

# Create DataLoader
batch_size = 16
train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
val_dataset = TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], val_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define the enhanced model
class EnhancedClassifier(nn.Module):
    def __init__(self, bert_model, hidden_dim, output_dim):
        super(EnhancedClassifier, self).__init__()
        self.bert = bert_model
        self.fc1 = nn.Linear(bert_model.config.hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_outputs.pooler_output  # Use the pooled output from BERT
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Parameters
hidden_dim = 256
output_dim = len(set(y_encoded))

# Initialize the model
model = EnhancedClassifier(bert_model, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}')

# Validation after training
model.eval()
val_preds = []
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)
        val_preds.extend(predicted.cpu().numpy())

# Calculate overall metrics
overall_accuracy = accuracy_score(val_labels, val_preds)
overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted')

# Calculate metrics for each class
precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(val_labels, val_preds, average=None)

# Map labels back to class names
class_names = label_encoder.inverse_transform([0, 1])

# Print overall metrics
print(f'Overall Accuracy: {overall_accuracy:.4f} (Overall text)')
print(f'Overall Precision: {overall_precision:.4f} (Overall text)')
print(f'Overall Recall: {overall_recall:.4f} (Overall text)')
print(f'Overall F1-Score: {overall_f1:.4f} (Overall text)')

# Print metrics for each class
for i, class_name in enumerate(class_names):
    class_indices = (val_labels == i)
    class_accuracy = accuracy_score(val_labels[class_indices], [val_preds[idx] for idx in range(len(val_preds)) if class_indices[idx]])
    print(f'Accuracy for {class_name} text: {class_accuracy:.4f} ({class_name} text)')
    print(f'Precision for {class_name} text: {precision_per_class[i]:.4f} ({class_name} text)')
    print(f'Recall for {class_name} text: {recall_per_class[i]:.4f} ({class_name} text)')
    print(f'F1 Score for {class_name} text: {f1_per_class[i]:.4f} ({class_name} text)')
