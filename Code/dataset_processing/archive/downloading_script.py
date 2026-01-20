# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")

# print("Path to dataset files:", path)

# from datasets import load_dataset

# ds = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("algord/fake-news")

# print("Path to dataset files:", path)

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("aadyasingh55/fake-news-classification")

# print("Path to dataset files:", path)

# from datasets import load_dataset
# import os

# # Load the dataset
# ds = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")

# # Define output directory
# output_dir = "Datasets-Huggingface"
# os.makedirs(output_dir, exist_ok=True)

# # Save each split (e.g., train, test, validation) as a separate CSV
# for split, dataset in ds.items():
#     dataset.to_csv(f"{output_dir}/{split}.csv")

# print(f"CSV files saved to {output_dir}")
