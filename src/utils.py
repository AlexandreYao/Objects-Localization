from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import numpy as np
import torch
import time


class EarlyStopper:
    """An early stopping mechanism for training loops.
    This class monitors the validation loss during training and triggers early stopping
    if the loss does not improve for a certain number of epochs.
    Parameters:
        patience (int): The number of epochs to wait before stopping if no improvement is seen.
            Default is 3.
        delta (float): The minimum change in validation loss to be considered as improvement.
            Default is 0.
    Attributes:
        patience (int): The number of epochs to wait before stopping if no improvement is seen.
        delta (float): The minimum change in validation loss to be considered as improvement.
        counter (int): The number of epochs since the last improvement in validation loss.
        best_loss (float): The best validation loss achieved so far.
        early_stop (bool): Whether to trigger early stopping or not.
    Methods:
        should_stop(val_loss): Checks if early stopping criteria are met based on the provided validation loss.
    """

    def __init__(self, patience=3, delta=0):
        """
        Initializes the EarlyStopper object with the specified patience and delta parameters.
        Args:
            patience (int): The number of epochs to wait before stopping if no improvement is seen.
                Default is 3.
            delta (float): The minimum change in validation loss to be considered as improvement.
                Default is 0.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def should_stop(self, val_loss):
        """
        Checks if early stopping criteria are met based on the provided validation loss.
        Args:
            val_loss (float): The validation loss obtained during training.
        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def train_classification_model(
    model,
    optimizer,
    criterion,
    train_dataloader,
    val_dataloader,
    num_epochs,
    device,
    scheduler=None,
    plot_figs=True,
    nb_batches_to_display=10,
    early_stopper=None,
    save_filepath=None
):
    """
    Train a classification model.
    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation set.
        num_epochs (int): Number of epochs for training.
        device (torch.device): Device to use for training (e.g., 'cpu', 'cuda').
        scheduler (optional): Learning rate scheduler.
        plot_figs (bool): Whether to plot training and validation curves.
        nb_batches_to_display (int): Number of batches to display during training.
        early_stopper (optional): Early stopping mechanism.
        save_filepath(optional): path to save the model
    Returns:
        tuple: Tuple containing the trained model and training history.
    """
    print(f"Train: model={type(model).__name__}, opt={type(optimizer).__name__}(lr={
          optimizer.param_groups[0]['lr']}), num_epochs={num_epochs}, device={device}\n")
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    start_time_sec = time.time()
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print("=" * 60)
        # TRAINING
        model.train()
        train_loss = 0.0
        num_train_correct = 0
        num_train_examples = 0
        for batch_index, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            num_train_correct += (predicted == targets).sum().item()
            num_train_examples += inputs.size(0)
            if (
                nb_batches_to_display > 0
                and (batch_index + 1) % nb_batches_to_display == 0
            ):
                print(
                    f"\tBatch {
                        batch_index+1}/{len(train_dataloader)}, loss: {train_loss / num_train_examples:.4f}"
                )
        train_acc = num_train_correct / num_train_examples
        train_loss = train_loss / len(train_dataloader.dataset)
        # VALIDATION
        model.eval()
        val_loss = 0.0
        num_val_correct = 0
        num_val_examples = 0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                num_val_correct += (predicted == targets).sum().item()
                num_val_examples += inputs.size(0)
        val_acc = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dataloader.dataset)
        print(
            f"Epoch {epoch}/{num_epochs}, train loss: {train_loss:.4f}, train acc: {
                train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}\n"
        )
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        # EARLY STOPPING
        if early_stopper is not None and early_stopper.should_stop(val_loss):
            print("Early stopping triggered.")
            break
    # END OF TRAINING
    end_time_sec = time.time()
    total_time_sec = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / num_epochs
    print(f"Time total:     {total_time_sec:.2f} sec")
    print(f"Time per epoch: {time_per_epoch_sec:.2f} sec\n")
    # PLOT CURVES
    if plot_figs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(history["loss"], label="Train Loss",
                     color="blue", marker="o")
        axes[0].plot(history["val_loss"],
                     label="Validation Loss", color="orange", marker="x")
        axes[0].set_xlabel("Number of Epochs", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title(
            "Training and Validation Loss Over Epochs", fontsize=14)
        axes[0].legend(loc="best", fontsize=12)
        axes[0].grid(True, linestyle="--", alpha=0.7)
        axes[1].plot(history["acc"], label="Train Accuracy",
                     color="blue", marker="o")
        axes[1].plot(history["val_acc"],
                     label="Validation Accuracy", color="orange", marker="x")
        axes[1].set_xlabel("Number of Epochs", fontsize=12)
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].set_title(
            "Training and Validation Accuracy Over Epochs", fontsize=14)
        axes[1].legend(loc="best", fontsize=12)
        axes[1].grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
    if save_filepath is not None:
        torch.save(model, save_filepath)
    return model, history


def evaluate_model(model, test_loader, device):
    """
    Evaluate the trained model on the test set and print the accuracy.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (torch.device): Device to use for evaluation (e.g., 'cpu', 'cuda').

    Returns:
        None
    """
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.numpy())
            all_targets.extend(targets.numpy())
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f'Accuracy on test set: {accuracy:.5f}')


def get_misclassified_samples(model, test_loader, device):
    """
    Evaluate the trained model on the test set and return a DataLoader with the misclassified samples.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (torch.device): Device to use for evaluation (e.g., 'cpu', 'cuda').

    Returns:
        torch.utils.data.DataLoader: A DataLoader containing the misclassified images, 
                                     their true labels, and the model's predictions.
    """
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            # Identify misclassified samples
            misclassified_indices = (
                predictions != targets).nonzero(as_tuple=True)[0]
            for idx in misclassified_indices:
                misclassified_images.append(images[idx])
                misclassified_labels.append(targets[idx])
                misclassified_predictions.append(predictions[idx])

    # Create a DataLoader from the misclassified samples
    misclassified_images_tensor = torch.stack(misclassified_images)
    misclassified_labels_tensor = torch.tensor(misclassified_labels)
    misclassified_predictions_tensor = torch.tensor(misclassified_predictions)

    # Combine targets and predictions into a single tensor
    combined_labels_tensor = torch.stack(
        (misclassified_labels_tensor, misclassified_predictions_tensor), dim=1)

    # Create a TensorDataset including images and the combined labels and predictions
    misclassified_dataset = TensorDataset(
        misclassified_images_tensor, combined_labels_tensor)
    misclassified_dataloader = DataLoader(
        misclassified_dataset, batch_size=32, shuffle=False)
    return misclassified_dataloader


def count_trainable_parameters(m):
    """
    Compte le nombre de paramètres entraînables dans un modèle PyTorch.
    Args:
        m (torch.nn.Module): Le modèle PyTorch dont les paramètres entraînables doivent être comptés.
    Returns:
        int: Le nombre total de paramètres entraînables dans le modèle.
    Example:
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 5)
        >>> num_trainable_params = count_trainable_parameters(model)
        >>> print(num_trainable_params)
    """
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def count_elements_per_class(image_folder):
    """
    Compte le nombre d'éléments par classe dans un dataset ImageFolder.
    Args:
        image_folder (torchvision.datasets.ImageFolder): Le dataset ImageFolder contenant les images et les classes.
    Returns:
        dict: Un dictionnaire où les clés sont les noms des classes et les valeurs sont le nombre d'éléments par classe.
    Example:
        >>> data_dir = 'path/to/your/data'
        >>> dataset = ImageFolder(root=data_dir)
        >>> class_counts = count_elements_per_class(dataset)
        >>> for class_name, count in class_counts.items():
        >>>     print(f"Classe: {class_name}, Nombre d'éléments: {count}")
    """
    class_counts = Counter()
    for _, target in image_folder.samples:
        class_name = image_folder.classes[target]
        class_counts[class_name] += 1
    return dict(class_counts)


def show_images(dataset, num_rows=4, num_cols=3, class_map=None):
    """
    Display a grid of images from a dataset.
    Args:
        dataset (Dataset): The dataset from which to display images. The dataset should return (image, label) tuples.
        num_rows (int, optional): The number of rows in the image grid. Must be greater than or equal to 1. Default is 4.
        num_cols (int, optional): The number of columns in the image grid. Must be greater than or equal to 1. Default is 3.
        class_map (dict, optional): A dictionary mapping class labels to class names. If None, labels are displayed as is. Default is None.
    Raises:
        AssertionError: If `num_rows` or `num_cols` is less than 1.
        AssertionError: If the dataset has fewer samples than `num_rows * num_cols`.
    Example:
        >>> from torchvision.datasets import CIFAR10
        >>> import torchvision.transforms as transforms
        >>> transform = transforms.Compose([transforms.ToTensor()])
        >>> dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        >>> class_map = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                         5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        >>> show_images(dataset, num_rows=4, num_cols=3, class_map=class_map)
    """
    nb_samples = len(dataset)
    num_samples_to_display = num_rows * num_cols
    assert (
        num_rows >= 1 and num_cols >= 1
    ), "Le nombre de colonnes et le nombre de lignes doivent être supérieurs ou égaux à 1"
    assert (
        nb_samples >= num_samples_to_display
    ), f"Trop peu de données à afficher. Il y a {nb_samples} données, vous souhaitez en afficher {num_samples_to_display} !"
    _, axes = plt.subplots(num_rows, num_cols, figsize=(
        num_cols * 2, num_rows * 2.5))
    for i, ith_dataset in enumerate(
        tqdm(np.random.randint(low=0, high=nb_samples, size=num_samples_to_display))
    ):
        image, label = dataset[ith_dataset]
        ii = i // num_cols
        jj = i % num_cols
        if image.shape[0] == 1:
            axes[ii, jj].imshow(
                image.numpy().transpose((1, 2, 0)), cmap="gray", interpolation="none"
            )
        else:
            axes[ii, jj].imshow(image.numpy().transpose((1, 2, 0)))
        axes[ii, jj].axis("off")
        axes[ii, jj].set_title(
            class_map[label] if class_map is not None else label)
    plt.tight_layout()
    plt.show()
