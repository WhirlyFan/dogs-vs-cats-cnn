import re
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def find_classes(directory: str):
  directory = Path(directory)
  class_names_set = set()
  files_by_class = {}

  for file_path in directory.glob("*.jpg"):
      # Extract the class name using regex
      match = re.match(r"(\w+)\.\d+\.jpg", file_path.name)
      if match:
          class_name = match.group(1)
      else:
          class_name = "no_class_name"

      if class_name not in files_by_class:
          files_by_class[class_name] = []
      files_by_class[class_name].append(file_path)
      class_names_set.add(class_name)

  class_names = sorted(class_names_set)  # Sort class names to ensure consistent indexing
  class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

  return class_names, class_to_idx

def plot_loss_curves(results: Dict[str, List[float]]):
  """Plots training curves of a results dictionary."""
  # Get the loss values of the results dictionary (training and test)
  loss = results["train_loss"]
  test_loss = results["test_loss"]

  #  Get the accuracy values of the results dictionary (training and test)
  accuracy = results["train_acc"]
  test_accuracy = results["test_acc"]

  # Figure out how many epochs there were
  epochs = range(len(results["train_loss"]))

  # Setup a plot
  plt.figure(figsize=(15, 7))

  # Plot the loss
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label="train_loss")
  plt.plot(epochs, test_loss, label="test_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  # Plot the accuracy
  plt.subplot(1, 2, 2)
  plt.plot(epochs, accuracy, label="train_accuracy")
  plt.plot(epochs, test_accuracy, label="test_accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()

  # Show the plot
  plt.show()
