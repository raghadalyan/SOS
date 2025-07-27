import numpy as np
import torch
from GameNetwork import GameNetwork, encode_action_50
from PUCTNode import PUCTNode

class PUCTPlayer:
    def __init__(self, game, model_path="game_network.pth", cpuct=1.0, simulations=100):
        self.game = game  # משחק ה-SOS
        self.cpuct = cpuct  # קבוע ה-CPUCT
        self.simulations = simulations  # כמות הסימולציות

        # טען את המודל המאומן
        self.network = GameNetwork(board_size=5)
        self.network.load_model(model_path)
        self.network.eval()

    def evaluate_state(self, state):
        """מקודד לוח לתוך הרשת ומחזיר policy ו-value"""
        size = len(state)
        encoded_rows = []
        for r in range(size):
            row_vals = []
            for cell in state[r]:
                if cell == 'S':
                    row_vals.append(1.0)
                elif cell == 'O':
                    row_vals.append(-1.0)
                else:
                    row_vals.append(0.0)
            encoded_rows.append(row_vals)

        board_array = np.array(encoded_rows, dtype=np.float32)
        board_tensor = torch.tensor(board_array.flatten(), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            policy_logits, value = self.network(board_tensor)
            policy = torch.softmax(policy_logits, dim=-1)

        return policy.squeeze(0).numpy(), value.item()

    def winning_move(self):
        """בודק אם יש מהלך מנצח מיידי – שמביא ניקוד, ומחזיר אותו אם קיים"""
        legal_moves = self.game.get_legal_moves()
        current_score = self.game.scores[self.game.current_player]

        for move in legal_moves:
            row, col, letter = move
            cloned_game = self.game.clone()
            cloned_game.make_move(row, col, letter)
            new_score = cloned_game.scores[self.game.current_player]

            if new_score > current_score:
                return move  # מצא מהלך שמביא ניקוד

        return None  # אין מהלך מנצח מיידי

    def search(self, root_state):
        """PUCT חיפוש מהשורש עם רשת"""
        root = PUCTNode(root_state)

        for _ in range(self.simulations):
            node = root

            # --- 1. SELECTION PHASE ---
            while True:
                # העתקת הלוח של הצומת הנוכחי
                temp_game = self.game.clone()
                temp_game.board = [r[:] for r in node.state]
                legal_moves = temp_game.get_legal_moves()

                # אם אין מהלכים חוקיים (מצב טרמינלי) או שהצומת לא מלא, עוצרים
                if not legal_moves or not node.is_fully_expanded(legal_moves):
                    break

                action, node = node.select(self.cpuct)
                # node אמור להיות ילד חוקי – אם הוא None, נצא (כדי להגן)
                if node is None:
                    break

            # --- 2. EXPANSION PHASE ---
            temp_game = self.game.clone()
            temp_game.board = [r[:] for r in node.state]
            legal_moves = temp_game.get_legal_moves()
            if legal_moves:
                policy, value = self.evaluate_state(node.state)

                for action in legal_moves:
                    row, col, letter = action
                    new_game = self.game.clone()
                    new_game.board = [r[:] for r in node.state]  # העתקת הלוח הנוכחי
                    new_game.make_move(row, col, letter)

                    move_index = encode_action_50(row, col, letter, self.game.size)
                    node.expand(action, new_game.board, policy[move_index])

                value = value if node.parent else 0
            else:
                # מצב טרמינלי – קביעת ערך לפי ניצחון
                winner = self.game.is_winner()
                if winner == "Player 1":
                    value = 1
                elif winner == "Player 2":
                    value = -1
                else:
                    value = 0

            # --- 3. BACKPROP PHASE ---
            while node:
                node.update(value)
                value = -value  # היפוך תור (משחק דו-צדדי)
                node = node.parent

        return root

    def get_best_action(self, state):
        """בודק קודם אם יש מהלך מנצח, אם לא – מפעיל PUCT ומחזיר את המהלך הטוב ביותר"""
        winning = self.winning_move()
        if winning:
            return winning  # מצאנו מהלך מנצח מיידי, לא צריך PUCT

        root = self.search(state)
        best_action, _ = root.select(cpuct=0)  # לבחור את המהלך הכי מבוקר (ללא חלק החקירה)
        return best_action
