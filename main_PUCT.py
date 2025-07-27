from SOS import SOSGame
from PUCTPlayer import PUCTPlayer

def play_against_puct():
    game = SOSGame(size=5)  # Initialize the SOS game
    ai_player = PUCTPlayer(game, model_path="game_network2.pth", simulations=100)
    
    print("Welcome to SOS Game! You are Player 1. The AI is Player 2.")
    game.display_board()
    
    while not game.is_terminal():
        if game.current_player == "Player 1":
            while True:
                try:
                    row = int(input("Enter row (0-4): "))
                    col = int(input("Enter col (0-4): "))
                    letter = input("Enter letter (S or O): ").upper()
                    if letter in ('S', 'O') and game.make_move(row, col, letter):
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Enter numbers for row/col and 'S' or 'O' for letter.")
        else:
            print("AI is making a move...")
            best_action = ai_player.get_best_action(game.board)
            row, col = best_action[:2]
            letter = best_action[2]
            game.make_move(row, col, letter)
        
        game.display_board()
    
    print("Game Over!")
    winner = game.is_winner()
    print(f"Winner: {winner}")

if __name__ == "__main__":
    play_against_puct()