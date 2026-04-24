# ================================
# SELF-PRUNING NEURAL NETWORK
# ================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ================================
# DEVICE SETUP
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# PRUNABLE LINEAR LAYER
# ================================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)      # [0,1]
        pruned_weights = self.weight * gates         # Apply pruning
        return torch.matmul(x, pruned_weights.t()) + self.bias


# ================================
# MODEL
# ================================
class PrunableNet(nn.Module):
    def __init__(self):
        super(PrunableNet, self).__init__()

        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# ================================
# DATA LOADING
# ================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# ================================
# SPARSITY LOSS
# ================================
def sparsity_loss(model):
    loss = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            loss += torch.sum(gates)
    return loss


# ================================
# EVALUATION FUNCTION
# ================================
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# ================================
# SPARSITY CALCULATION
# ================================
def calculate_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            pruned += (gates < threshold).sum().item()

    return 100 * pruned / total


# ================================
# PLOT GATE DISTRIBUTION
# ================================
def plot_gates(model):
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
            all_gates.extend(gates.flatten())

    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.show()


# ================================
# TRAINING FUNCTION
# ================================
def train_model(lambda_val, epochs=5):
    model = PrunableNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            classification_loss = criterion(outputs, labels)
            sparse_loss = sparsity_loss(model)

            loss = classification_loss + lambda_val * sparse_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Lambda {lambda_val} | Epoch {epoch+1} | Loss: {running_loss:.4f}")

    acc = evaluate(model)
    sparsity = calculate_sparsity(model)

    print(f"\nLambda {lambda_val} Results:")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%\n")

    plot_gates(model)

    return acc, sparsity


# ================================
# MAIN EXECUTION
# ================================
if __name__ == "__main__":

    lambda_values = [1e-5, 1e-4, 1e-3]

    results = []

    for lam in lambda_values:
        acc, sp = train_model(lam)
        results.append((lam, acc, sp))

    print("\nFINAL RESULTS:")
    print("Lambda\tAccuracy\tSparsity")
    for r in results:
        print(f"{r[0]}\t{r[1]:.2f}%\t\t{r[2]:.2f}%")