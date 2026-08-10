# ============================================================
# PART 2
# SIMPLE NEURAL NETWORK FOR MNIST CLASSIFICATION
#
# Optimizers:
# 1. SGD with Momentum
# 2. Adagrad
# 3. RMSprop
# 4. Adam
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import time
import copy


# ------------------------------------------------------------
# 1. DEVICE
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ------------------------------------------------------------
# 2. HYPERPARAMETERS
# ------------------------------------------------------------

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10

# These values are kept IDENTICAL for every optimizer.
print("\nHyperparameters:")
print("Batch size    :", BATCH_SIZE)
print("Learning rate :", LEARNING_RATE)
print("Epochs        :", EPOCHS)
print("Hidden units  :", HIDDEN_SIZE)


# ------------------------------------------------------------
# 3. LOAD AND PREPROCESS MNIST
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),

    # Normalize MNIST pixels
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])


train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# ------------------------------------------------------------
# 4. TRAIN / VALIDATION SPLIT
# ------------------------------------------------------------

train_size = 50000
validation_size = 10000

train_dataset, validation_dataset = random_split(
    train_dataset,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)


# ------------------------------------------------------------
# 5. DATA LOADERS
# ------------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nDataset:")
print("Training   :", len(train_dataset))
print("Validation :", len(validation_dataset))
print("Testing    :", len(test_dataset))


# ------------------------------------------------------------
# 6. DISPLAY SAMPLE IMAGE
# ------------------------------------------------------------

images, labels = next(iter(train_loader))

plt.figure(figsize=(5, 5))

plt.imshow(
    images[0].squeeze(),
    cmap="gray"
)

plt.title(f"Digit: {labels[0].item()}")
plt.axis("off")

plt.show()


# ------------------------------------------------------------
# 7. DEFINE SIMPLE NEURAL NETWORK
# ------------------------------------------------------------

class SimpleMNISTNet(nn.Module):

    def __init__(self):

        super(SimpleMNISTNet, self).__init__()

        self.network = nn.Sequential(

            # Input layer
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),

            # Hidden layer activation
            nn.ReLU(),

            # Output layer
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

        )

    def forward(self, x):

        # Flatten 28x28 image into 784 values
        x = x.view(x.size(0), -1)

        return self.network(x)


# IMPORTANT:
# We DO NOT put Softmax inside the model.
#
# CrossEntropyLoss internally applies the appropriate
# log-softmax operation to the output logits.


# ------------------------------------------------------------
# 8. LOSS FUNCTION
# ------------------------------------------------------------

criterion = nn.CrossEntropyLoss()


# ------------------------------------------------------------
# 9. CREATE OPTIMIZERS
# ------------------------------------------------------------

def create_optimizer(model, optimizer_name):

    if optimizer_name == "SGD with Momentum":

        return optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=0.9
        )

    elif optimizer_name == "Adagrad":

        return optim.Adagrad(
            model.parameters(),
            lr=LEARNING_RATE
        )

    elif optimizer_name == "RMSprop":

        return optim.RMSprop(
            model.parameters(),
            lr=LEARNING_RATE
        )

    elif optimizer_name == "Adam":

        return optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )


# ------------------------------------------------------------
# 10. TRAINING FUNCTION
# ------------------------------------------------------------

def train_one_epoch(
    model,
    optimizer,
    criterion,
    loader
):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Statistics
        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = total_loss / total

    accuracy = 100 * correct / total

    return average_loss, accuracy


# ------------------------------------------------------------
# 11. VALIDATION FUNCTION
# ------------------------------------------------------------

def evaluate(
    model,
    criterion,
    loader
):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / total

    accuracy = 100 * correct / total

    return average_loss, accuracy


# ------------------------------------------------------------
# 12. TRAIN EACH OPTIMIZER
# ------------------------------------------------------------

optimizer_names = [
    "SGD with Momentum",
    "Adagrad",
    "RMSprop",
    "Adam"
]

results = {}

for optimizer_name in optimizer_names:

    print("\n" + "=" * 70)
    print("Training using:", optimizer_name)
    print("=" * 70)

    # Create a fresh model for each optimizer.
    # This is important for a fair comparison.
    model = SimpleMNISTNet().to(device)

    optimizer = create_optimizer(
        model,
        optimizer_name
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "time": []
    }

    start_time = time.time()

    for epoch in range(EPOCHS):

        epoch_start = time.time()

        # Training
        train_loss, train_acc = train_one_epoch(
            model,
            optimizer,
            criterion,
            train_loader
        )

        # Validation
        val_loss, val_acc = evaluate(
            model,
            criterion,
            validation_loader
        )

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        history["time"].append(epoch_time)

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.2f}% "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc:.2f}%"
        )

    total_time = time.time() - start_time

    history["total_time"] = total_time

    # Save final model
    history["model"] = copy.deepcopy(model)

    results[optimizer_name] = history

    print(
        f"\nTotal training time: "
        f"{total_time:.2f} seconds"
    )


# ------------------------------------------------------------
# 13. COMPARE FINAL RESULTS
# ------------------------------------------------------------

print("\n")
print("=" * 90)
print("FINAL OPTIMIZER COMPARISON")
print("=" * 90)

print(
    f"{'Optimizer':<22}"
    f"{'Train Acc':>12}"
    f"{'Val Acc':>12}"
    f"{'Train Loss':>14}"
    f"{'Val Loss':>12}"
    f"{'Time(s)':>12}"
)

print("-" * 90)

for name, history in results.items():

    print(
        f"{name:<22}"
        f"{history['train_acc'][-1]:>11.2f}%"
        f"{history['val_acc'][-1]:>11.2f}%"
        f"{history['train_loss'][-1]:>14.4f}"
        f"{history['val_loss'][-1]:>12.4f}"
        f"{history['total_time']:>12.2f}"
    )


# ------------------------------------------------------------
# 14. TRAINING ACCURACY VS EPOCH
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for name, history in results.items():

    plt.plot(
        range(1, EPOCHS + 1),
        history["train_acc"],
        marker="o",
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Training Accuracy (%)")
plt.title("Training Accuracy vs Epoch")

plt.legend()
plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 15. VALIDATION ACCURACY VS EPOCH
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for name, history in results.items():

    plt.plot(
        range(1, EPOCHS + 1),
        history["val_acc"],
        marker="o",
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy vs Epoch")

plt.legend()
plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 16. TRAINING LOSS VS EPOCH
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for name, history in results.items():

    plt.plot(
        range(1, EPOCHS + 1),
        history["train_loss"],
        marker="o",
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")

plt.legend()
plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 17. VALIDATION LOSS VS EPOCH
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for name, history in results.items():

    plt.plot(
        range(1, EPOCHS + 1),
        history["val_loss"],
        marker="o",
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs Epoch")

plt.legend()
plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 18. IDENTIFY FASTEST CONVERGING OPTIMIZER
# ------------------------------------------------------------

# Here we define "fastest convergence" as the optimizer
# reaching the highest validation accuracy by the end of
# training while also having a low validation loss.

final_val_acc = {
    name: history["val_acc"][-1]
    for name, history in results.items()
}

final_val_loss = {
    name: history["val_loss"][-1]
    for name, history in results.items()
}

fastest_optimizer = max(
    final_val_acc,
    key=final_val_acc.get
)

best_loss_optimizer = min(
    final_val_loss,
    key=final_val_loss.get
)


print("\n" + "=" * 70)
print("OBSERVATION")
print("=" * 70)

print(
    "Highest final validation accuracy:",
    fastest_optimizer
)

print(
    "Lowest final validation loss:",
    best_loss_optimizer
)


# ------------------------------------------------------------
# 19. AUTOMATIC COMMENT / JUSTIFICATION
# ------------------------------------------------------------

print("""
------------------------------------------------------------
JUSTIFICATION
------------------------------------------------------------

Adam generally converges faster in this experiment because it
combines the advantages of momentum and adaptive learning rates.

SGD with Momentum accelerates learning in consistent gradient
directions and reduces oscillations.

Adagrad adapts the learning rate for every parameter, but its
learning rate can become very small as training progresses.

RMSprop also uses adaptive learning rates and usually converges
faster than basic SGD for this type of neural network.

Adam combines momentum-like first-moment estimation with
second-moment adaptive learning rates. Therefore, it usually
reaches a good validation accuracy in fewer epochs and provides
stable convergence.

The actual fastest optimizer may vary depending on the random
initialization, hardware, learning rate, batch size and number
of epochs. The plots and final metrics should be used to verify
the observation for the current run.
------------------------------------------------------------
""")


# ------------------------------------------------------------
# 20. OPTIONAL: TEST THE BEST MODEL
# ------------------------------------------------------------

best_model = results[fastest_optimizer]["model"]

test_loss, test_accuracy = evaluate(
    best_model,
    criterion,
    test_loader
)

print("\n" + "=" * 70)
print("BEST MODEL TEST RESULT")
print("=" * 70)

print("Optimizer :", fastest_optimizer)
print(f"Test Loss : {test_loss:.4f}")
print(f"Test Acc  : {test_accuracy:.2f}%")