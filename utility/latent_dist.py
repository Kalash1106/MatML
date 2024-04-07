import os
import random
import torch

def folder_id_dict(folder_path):
  base_name = os.path.basename(folder_path)
  if base_name == 'crack_defective': return 0
  elif base_name == 'crack_non_defective': return 1
  elif base_name == 'ksdd_defective': return 2
  elif base_name == 'ksdd_non_defective': return 3

def get_random_images(root_dir, num_images_per_folder=2):
  """
  This function samples num_images_per_folder random images from each subdirectory within the root_dir without creating a complete list of images.

  Args:
      root_dir: Path to the root directory containing subfolders with images.
      num_images_per_folder: Number of images to sample from each subfolder (default: 2).

  Returns:
      A list of tuples, where each tuple contains the folder path and the list of sampled image paths.
  """
  sampled_images = []
  for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
      sampled_folder_images = []
      for _ in range(num_images_per_folder):
        # Get a random file from the directory
        filename = random.choice(os.listdir(folder_path))
        # Check if it's a file (prevents non-image files)
        if os.path.isfile(os.path.join(folder_path, filename)):
          sampled_folder_images.append(os.path.join(folder_path, filename))
      if sampled_folder_images:
        sampled_images.append((folder_id_dict(folder_path), sampled_folder_images))
  return sampled_images


def normalized_distance(x, y):
    """
    This function computes the normalized Euclidean distance between two tensors.

    Args:
        x: A PyTorch tensor.
        y: A PyTorch tensor with the same shape as x.

    Returns:
        A float representing the normalized Euclidean distance between x and y.
    """
    # Calculate difference
    diff = x - y

    # L2 norm (Euclidean distance)
    norm = torch.linalg.norm(diff)

    # Normalize by the norm of x (assuming non-zero norm)
    if x.norm() != 0:
        normalized_distance = norm / x.norm()
    else:
        normalized_distance = float('inf')  # Handle potential division by zero

    return normalized_distance

def embedding_dist(model, img1, img2, transform = None, embed_dim = 512):
    if transform is not None:
        from PIL import Image
        image1 = transform(Image.open(img1))
        image2 = transform(Image.open(img2))
    
    with torch.no_grad():
        embed1 = model(image1.unsqueeze(0)).reshape([embed_dim])
        embed2 = model(image2.unsqueeze(0)).reshape([embed_dim])

    return normalized_distance(embed1, embed2)


