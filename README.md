# 🧠 Self-Pruning Neural Network

## 📌 Overview
This project implements a neural network that learns to prune its own weights during training using learnable gates and L1 regularization.

---

## 🚀 Features
- Custom PrunableLinear layer
- Learnable gating mechanism
- Automatic pruning during training
- Sparsity vs accuracy analysis

---

## ⚙️ How it Works
Each weight has a gate:

gate = sigmoid(gate_scores)  
pruned_weight = weight × gate  

If gate → 0 → weight removed

---

## 📊 Results
| Lambda | Accuracy | Sparsity |
|--------|---------|----------|
| 1e-5   | 78.2    | 12.5     |
| 1e-4   | 75.6    | 34.8     |
| 1e-3   | 66.3    | 68.9     |

---

## ▶️ Run
pip install torch torchvision matplotlib  
python self_pruning_network.py  

---

## 👨‍💻 Author
Karthik Kumar
