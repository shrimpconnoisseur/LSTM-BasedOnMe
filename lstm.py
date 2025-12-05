import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class MessageDataset(Dataset):
  def __init__(self, json_file, seq_length=100):
    self.seq_length = seq_length
    
    with open(json_file, 'r', encoding='utf-8') as f:
      self.data = json.load(f)
      
    texts = [entry["Contents"] for entry in self.data if "Contents" in entry]
    
    self.text = "\n".join(texts)
    
    self.chars = sorted(list(set(self.text)))
    self.stoi = {ch: i for i, ch in enumerate(self.chars)}
    self.itos = {i: ch for i, ch in enumerate(self.chars)}
    self.vocab_size = len(self.chars)
    
    self.data = torch.tensor([self.stoi[c] for c in self.text])
    
  def __len__(self):
    return len(self.data) - self.seq_length
  
  def __getitem__(self, idx):
    seq = self.data[idx:idx+self.seq_length]
    target = self.data[idx+1:idx+self.seq_length+1]
    return seq, target
  

class LSTMModel(nn.Module):
  def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2):
    super(LSTMModel, self).__init__()
    
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, vocab_size)
    
  def forward(self, x, hidden=None):
    x = self.embedding(x)
    out, hidden = self.lstm(x, hidden)
    out = self.fc(out)
    return out, hidden
  
def generate(model, stoi, itos, start_text="", length=200, temperature=0.8):
  model.eval()
  idx_input = torch.tensor([stoi[c] for c in start_text], dtype=torch.long).unsqueeze(0)
  
  hidden = None
  output_text = start_text
  
  with torch.no_grad():
    for _ in range(length):
      logits, hidden = model(idx_input, hidden)
      logits = logits[:, -1, :] / temperature
      probs = torch.softmax(logits, dim=-1)
      
      next_idx = torch.multinomial(probs, num_samples=1).item()
      next_char = itos[next_idx]
      output_text += next_char
      
      idx_input = torch.tensor([[next_idx]])
      
  return output_text

def train_model(json_file, epochs=15, batch_size=128, lr=1e-3, seq_length=75, save_path="lstm_model-2.pth"): # ---------------
  dataset = MessageDataset(json_file, seq_length)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  
  model = LSTMModel(dataset.vocab_size)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()
  
  for epoch in range(epochs):
    total_loss = 0
    for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
      optimizer.zero_grad()
      logits, _ = model(x)
      loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
    
  torch.save({
      'model_state_dict': model.state_dict(),
      'stoi': dataset.stoi,
      'itos': dataset.itos
  }, save_path)
  
  print(f"Model saved to {save_path}")
  
def test(save_path="lstm_model-2.pth"): # ---------------
  checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
  stoi, itos = checkpoint['stoi'], checkpoint['itos']
  vocab_size = len(stoi)
  
  model = LSTMModel(vocab_size)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()
  
  print("Type in a prompt to generate a predicted response.\n*Type 'exit' to quit.*\n")
  
  while True:
    prompt = input("Prompt: ")
    if prompt.lower() == 'exit':
      break
    if prompt.strip() == "":
      continue
    
    response = generate(model, stoi, itos, start_text=prompt, length=200)
    print("Output:", response[len(prompt):].strip(), "\n")
    
if __name__ == "__main__":
  import argparse
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", action="store_true")
  parser.add_argument("--test", action="store_true")
  parser.add_argument("--json", type=str, default="merged_messages.json", help="Path to JSON file for training")
  args = parser.parse_args()
  
  if args.train:
    if not args.json:
      print("Error: need --json <path>")
    else:
      train_model(args.json)
      
  if args.test:
    test()
    
# Model 1 Params:
# Epochs: 5
# Batch Size: 32
# Learning Rate: 3e-4
# Sequence Length: 100
# Hidden Dim: 256
# Embedding Dim: 128
# Layers: 2
# Time Elapsed: 2 hours 34 minutes

# Model 2 Params:
# Epochs: 10
# Batch Size: 64
# Learning Rate: 1e-3
# Sequence Length: 75
# Hidden Dim: 128
# Embedding Dim: 128
# Layers: 2
# Time Elapsed: 54 minutes

# Model "speed" Params:
# Epochs: 10
# Batch Size: 128
# Learning Rate: 1e-3
# Sequence Length: 75
# Hidden Dim: 128
# Embedding Dim: 128
# Layers: 2
# Time Elapsed: i forgot lol