# SOS – Reinforcement Learning Agent

This project implements a Reinforcement Learning agent that learns to play the classic SOS game using self-play and policy improvement techniques. The agent uses the PUCT (Predictor + UCT) algorithm to improve decision-making over time and is trained using neural networks in PyTorch.

## 🎯 Project Goals

- Develop an intelligent agent that can learn and play SOS optimally.
- Use Reinforcement Learning with self-play to improve strategy over time.
- Implement and test the PUCT algorithm.
- Visualize training progress and evaluate model performance.

## 🧠 Key Features

- **PUCT Algorithm**: Efficient tree-based decision-making.
- **Self-play Training**: The model improves by playing against itself.
- **Neural Networks**: Trained using PyTorch (`GameNetwork.py`, `train_with_PUCT.py`).
- **CSV Logging**: Game states and moves are saved for evaluation (`self_play_data2.csv`).

## 📁 Project Structure

sos/
├── GameNetwork.py # Neural network architecture
├── GameNetwork2.py # Alternate model version
├── main_PUCT.py # Entry point for training
├── train_with_PUCT.py # Self-play training script
├── PUCTPlayer.py # RL agent using PUCT
├── PUCTNode.py # Tree node structure
├── self_play_data2.csv # Collected game data
└── game_network2.pth # Trained model file

## 🧪 How to Run

1. Install dependencies:
   ```bash
   pip install torch numpy
