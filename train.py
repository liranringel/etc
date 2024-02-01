import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')

def train(args, model, n_timesteps, num_classes, train_dl, val_dl):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lowest_val_loss = float('inf')
    num_epochs_without_improvement = 0
    best_model = None
    epoch = 1

    while True:
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            res = model(x)
            all_scores = res['all_scores']
            all_is_correct_estimation = res['all_is_correct_estimation']
            y_expanded = y.view(-1, 1).expand(-1, n_timesteps)
            is_correct = y_expanded == all_scores.argmax(dim=-1)
            is_correct_loss = nn.BCELoss()(all_is_correct_estimation, is_correct.float())
            classification_loss = criterion(all_scores.reshape(-1, num_classes), y_expanded.reshape(-1))
            loss = classification_loss + args.is_correct_loss_factor * is_correct_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses = []
        correct = 0
        total = 0
        halt_timesteps = []

        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                res = model(x)
                all_scores = res['all_scores']
                all_is_correct_estimation = res['all_is_correct_estimation']
                y_expanded = y.view(-1, 1).expand(-1, n_timesteps)
                is_correct = y_expanded == all_scores.argmax(dim=-1)
                is_correct_loss = nn.BCELoss()(all_is_correct_estimation, is_correct.float())
                classification_loss = criterion(all_scores.reshape(-1, num_classes), y_expanded.reshape(-1))
                loss = classification_loss + args.is_correct_loss_factor * is_correct_loss
                losses.append(loss.item())
                scores = res['scores']
                halt_timesteps.append(res['halt_timesteps'])
                _, predicted = torch.max(scores.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        val_loss = np.mean(losses)
        acc = 100 * correct / total
        mean_halt_timesteps = torch.cat(halt_timesteps).float().mean().item()
        print(f'Epoch {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc:.2f}%, Mean Halt Timesteps: {mean_halt_timesteps:.2f}')

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            num_epochs_without_improvement = 0
            best_model = copy.deepcopy(model.to('cpu'))
        else:
            num_epochs_without_improvement += 1
            if num_epochs_without_improvement >= 30:
                break
        epoch += 1

    return best_model
