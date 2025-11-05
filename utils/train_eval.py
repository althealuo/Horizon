import torch
import torch.nn as nn

def train(model, train_loader, criterion, optimizer, device): 
    model.train()
    train_loss = 0.0
    correct = 0
    total = len(train_loader)

    for seq_inputs, static_inputs, labels in train_loader:
        seq_inputs, static_inputs, labels = seq_inputs.to(device), static_inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(seq_inputs, static_inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        train_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()

    avg_loss = train_loss / total
    accuracy = correct / total
    return accuracy, avg_loss


def test(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total_loss = 0
    total = len(test_loader.dataset)
    
    with torch.no_grad():
        for seq_inputs, static_inputs, labels in test_loader:
            seq_inputs, static_inputs, labels = seq_inputs.to(device), static_inputs.to(device), labels.to(device)
            
            logits = model(seq_inputs, static_inputs)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)


    accuracy = correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss

def train_and_evaluate(model, train_loader, test_loaders, criterion, optimizer, device, epochs):
    test_loader, test_loader_h1, test_loader_h6 = test_loaders
    train_loss_prog, train_acc_prog = [], []

    test_loss_prog, test_acc_prog = [], []

    test_acc_h1_prog, test_loss_h1_prog = [], []
    test_acc_h6_prog, test_loss_h6_prog = [], []

    epochs_without_improvement = 0 # for early stopping
    best_loss = float('inf')
    PATIENCE = 5
    final_epoch = epochs

    for epoch in range(epochs):
        train_acc, train_loss = train(model, train_loader, criterion, optimizer, device)

        test_acc, test_loss = test(model, test_loader, criterion, device)
        test_acc_h1, test_loss_h1 = test(model, test_loader_h1, criterion, device)
        test_acc_h6, test_loss_h6 = test(model, test_loader_h6, criterion, device)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Loss: {test_loss:.4f} | overall: {test_acc:.4f} | H1 {test_acc_h1:.4f} | H6 {test_acc_h6:.4f}")

        train_acc_prog.append(train_acc)
        train_loss_prog.append(train_loss)

        test_loss_prog.append(test_loss)
        test_acc_prog.append(test_acc)

        test_acc_h1_prog.append(test_acc_h1)
        test_loss_h1_prog.append(test_loss_h1)

        test_acc_h6_prog.append(test_acc_h6)
        test_loss_h6_prog.append(test_loss_h6)

        # early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement > PATIENCE: 
            print(f"Early stopping triggered: epoch {epoch+1} best_loss {best_loss:.4f}")
            final_epoch = epoch+1
            break

    return {
        "train_loss_prog": train_loss_prog,
        "train_acc_prog": train_acc_prog,
        "test_loss_prog": test_loss_prog,
        "test_acc_prog": test_acc_prog,
        "test_acc_h1_prog": test_acc_h1_prog,
        "test_loss_h1_prog": test_loss_h1_prog,
        "test_acc_h6_prog": test_acc_h6_prog,
        "test_loss_h6_prog": test_loss_h6_prog,
        "final_epoch": final_epoch
    }