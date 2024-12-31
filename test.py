import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import config
from train import SiameseNetwork, SiameseDataset

# Define device (CUDA if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_optimal_threshold_roc(model, dataloader):
    """Calculates the optimal threshold using ROC curve analysis."""
    distances = []  # Euclidean distances
    true_labels = []  # Actual labels

    with torch.no_grad():
        for data in dataloader:
            x0, x1, label = data
            x0, x1, label = x0.to(device), x1.to(device), label.to(device)

            # Forward pass
            output1, output2 = model(x0, x1)

            # Compute Euclidean distance
            euclidean_distance = F.pairwise_distance(output1, output2)

            distances.append(euclidean_distance.cpu().numpy())  # Store distances
            true_labels.append(label.cpu().numpy())  # Store true labels

    # Flatten the lists of distances and true_labels
    distances = np.concatenate(distances)
    true_labels = np.concatenate(true_labels)

    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, distances, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Find threshold with the best TPR/FPR trade-off
    optimal_idx = np.argmax(tpr - fpr)  # Optimal threshold where TPR is high and FPR is low
    optimal_threshold = thresholds[optimal_idx]

    print(f"Optimal Threshold: {optimal_threshold}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"True Positive Rate at Optimal Threshold: {tpr[optimal_idx]:.4f}")
    print(f"False Positive Rate at Optimal Threshold: {fpr[optimal_idx]:.4f}")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label="Optimal Threshold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()

    return optimal_threshold

def main():
    # Load the test dataset
    test_dataset = SiameseDataset(
        csv_file=config.testing_csv,
        image_dir=config.testing_dir,
        transform=transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])
    )

    test_dataloader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=True)

    # Load the trained model
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load("C:/Users/vijay/Desktop/Code/Signature_Verification v2/model_checkpoints/best_model.pth"))
    model.eval()

    # Calculate the optimal threshold using the ROC curve
    if hasattr(config, "optimal_threshold") and config.optimal_threshold is not None:
        print(f"Using pre-calculated threshold from config: {config.optimal_threshold}")
        optimal_threshold = config.optimal_threshold
    else:
        print("Calculating the optimal threshold...")
        optimal_threshold = calculate_optimal_threshold_roc(model, test_dataloader)

    # Testing the model with the calculated threshold
    print("\nTesting the model with the optimal threshold...")

    correct_predictions = 0
    total_samples = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0

    all_true_labels = []
    all_predictions = []

    for i, data in enumerate(test_dataloader, 0):
        x0, x1, label = data
        concat = torch.cat((x0, x1), 0)
        output1, output2 = model(x0.to(device), x1.to(device))

        # Compute Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Determine the prediction based on the threshold
        prediction = 1 if euclidean_distance.item() > optimal_threshold else 0

        # Store true labels and predictions for confusion matrix
        actual_label = int(label.item())

        all_true_labels.append(actual_label)
        all_predictions.append(prediction)

        # Calculate accuracy and other metrics
        if prediction == actual_label:
            correct_predictions += 1
            if actual_label == 1:
                true_positives += 1
            else:
                true_negatives += 1
        elif prediction == 1 and actual_label == 0:
            false_positives += 1
        elif prediction == 0 and actual_label == 1:
            false_negatives += 1

        total_samples += 1

    # Calculate and print accuracy after loop
    accuracy = (correct_predictions / total_samples) * 100
    false_positive_rate = (false_positives / total_samples) * 100
    false_negative_rate = (false_negatives / total_samples) * 100

    # Display Results
    print("\n--- Final Metrics ---")
    print(f"Total Samples = {total_samples} and Correct Predictions = {correct_predictions}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"False Positive Rate: {false_positive_rate:.2f}%")
    print(f"False Negative Rate: {false_negative_rate:.2f}%")

    print(f"\nFinal Optimal Threshold: {optimal_threshold}")

    # Confusion Matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    print(f"\nConfusion Matrix:\n{cm}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Forged', 'Original'])
    plt.yticks(tick_marks, ['Forged', 'Original'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == '__main__':
    main()
