import csv
import os
import numpy as np
from SOS import SOSGame
from PUCTPlayer import PUCTPlayer
from GameNetwork import encode_action_50

def self_play(num_games=1000, output_csv="self_play_data2.csv"):
    # בדוק אם הקובץ קיים, אחרת כתוב כותרת
    write_header = not os.path.exists(output_csv)
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["board_str", "pi_vector", "selected_move", "outcome"])
        
        for game_num in range(num_games):
            print(f"Game {game_num + 1}/{num_games}")
            game = SOSGame(size=5)
            ai_player = PUCTPlayer(game, model_path="game_network.pth", simulations=100)

            game_history = []

            while not game.is_terminal():
                # חיפוש עם PUCT
                root = ai_player.search(game.board)
                
                # חשב את פאי מתוך ביקורי הצמתים (וקטור בגודל 50)
                total_N = sum(child.N for child in root.children.values())
                pi_vector = np.zeros(50)
                for action, child in root.children.items():
                    row, col, letter = action
                    idx = encode_action_50(row, col, letter, game.size)
                    pi_vector[idx] = child.N / total_N

                # בחר מהלך לפי ההתפלגות (sampling)
                actions = list(root.children.keys())
                probs = np.array([child.N for child in root.children.values()])
                probs = probs / probs.sum()
                selected_action = actions[np.random.choice(len(actions), p=probs)]

                # שמור ללוג
                board_str = ''.join([''.join(r) for r in game.board])
                game_history.append({
                    "board_str": board_str,
                    "pi": pi_vector.tolist(),
                    "move": selected_action
                })

                # בצע את המהלך הנבחר
                row, col, letter = selected_action
                game.make_move(row, col, letter)

            # תוצאה סופית לאחר המשחק
            winner = game.is_winner()
            if winner == "Player 1":
                outcome = 1
            elif winner == "Player 2":
                outcome = -1
            else:
                outcome = 0

            # עדכן את כל ההיסטוריה עם התוצאה ושמור מיידית ל-CSV
            for step in game_history:
                row, col, letter = step["move"]
                writer.writerow([
                    step["board_str"],
                    step["pi"],
                    (row, col, letter),
                    outcome
                ])
            f.flush()  # שמירה מיידית לדיסק אחרי כל משחק

    print(f"Saved data to {output_csv}")

if __name__ == "__main__":
    self_play(num_games=1500)
