import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd

############################################
# רשת עצבית עם שני ראשים: Policy + Value
############################################
class GameNetwork(nn.Module):
    def __init__(self, board_size=5):
        super(GameNetwork, self).__init__()
        self.board_size = board_size
        self.input_dim = board_size * board_size  # Flattened board input
        
        # שכבות משותפות
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Value Head (Win probability / game outcome)
        self.value_head = nn.Linear(128, 1)
        
        # Policy Head עם 50 אפשרויות (25 תאים × 2 אותיות)
        self.policy_head = nn.Linear(128, 2 * board_size * board_size)  # =50 ל-board_size=5

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Value Head => [-1..1] עם tanh
        value = self.value_head(x)
        value = torch.tanh(value)
        
        # Policy Head => logits בגודל 50 (ללא softmax)
        policy_logits = self.policy_head(x)
        
        return policy_logits, value

    def save_model(self, file_path="game_network2.pth"):
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path="game_network2.pth"):
        self.load_state_dict(torch.load(file_path, map_location="cpu"))
        self.eval()
        print(f"Model loaded from {file_path}")


############################################
# encode_board
############################################
def encode_board(board_str, size=5):
    rows = board_str.split("|")
    board_array = np.zeros((size, size), dtype=np.float32)

    for r in range(size):
        row_cells = rows[r]
        for c in range(size):
            cell = row_cells[c]
            if cell == 'S':
                board_array[r, c] = 1.0
            elif cell == 'O':
                board_array[r, c] = -1.0
            else:
                board_array[r, c] = 0.0

    return board_array.flatten()

############################################
# encode_player
############################################
def encode_player(player):
    if player == "Player 1":
        return 1
    elif player == "Player 2":
        return -1
    else:
        return 0

############################################
# encode_action_50
############################################
def encode_action_50(row, col, letter, size=5):
    letter_idx = 0 if letter == 'S' else 1
    return (row * size + col) * 2 + letter_idx

############################################
# train_network
############################################
def train_network(csv_file, epochs=50, learning_rate=0.0005):
    df = pd.read_csv(csv_file)
    
    X_list = []        
    Y_policy_list = [] # אינדקס [0..49]
    Y_value_list = []  # ערך ב-[-1..1]
    
    for _, row_data in df.iterrows():
        board_str = row_data["board_str"]
        winner = row_data["winner"]
        
        # קידוד הלוח
        board_encoded = encode_board(board_str)
        
        # קידוד המנצח
        value_encoded = encode_player(winner)
        
        # נקרא את העמודות row, col, letter ונמפה לאינדקס 0..49
        r = int(row_data["row"])
        c = int(row_data["col"])
        letter = row_data["letter"]  # 'S' או 'O'
        move_index = encode_action_50(r, c, letter)
        
        X_list.append(board_encoded)
        Y_policy_list.append(move_index)
        Y_value_list.append([value_encoded])

    # המרה ל טנסורים
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y_policy = torch.tensor(Y_policy_list, dtype=torch.long)
    Y_value = torch.tensor(Y_value_list, dtype=torch.float32)
    
    # הגדרת מודל ואופטימייזר
    model = GameNetwork(board_size=5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        policy_logits, value_pred = model(X)
        
        loss_p = criterion_policy(policy_logits, Y_policy.squeeze())
        loss_v = criterion_value(value_pred, Y_value)
        loss = loss_p + loss_v
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}/{epochs} - Total Loss: {loss.item():.4f} (Policy={loss_p.item():.4f}, Value={loss_v.item():.4f})")
    
    model.save_model()
    print("Training completed!")


if __name__ == "__main__":
    train_network("self_play_data.csv", epochs=100)
    df = pd.read_csv("self_play_data.csv")
    print("Total games:", df["game_num"].nunique())
