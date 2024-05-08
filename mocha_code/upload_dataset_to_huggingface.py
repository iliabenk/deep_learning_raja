import pandas as pd
from datasets import Dataset, Features, Value, Image, load_dataset
import os
from huggingface_hub import HfApi

def gen_csv_to_upload(path, out_path):
    df = pd.read_csv(path)

    image_files = [f"{i:06d}.jpg" for i in range(len(df))]

    df["file_name"] = image_files

    # Move the 'text' column to be the first column
    columns = ['file_name'] + [col for col in df.columns if col != 'file_name']
    df = df[columns]

    df.to_csv(out_path, index=False)

def gen_data_for_hf(csv_path, images_dir, output_dir):
    # Load the CSV mapping into a Pandas DataFrame
    df = pd.read_csv(csv_path)

    # Create a dictionary for Dataset creation
    data_dict = {
        "image": [f"{os.path.join(images_dir, filename)}" for filename in df["image"]],
        "text": df["text"].tolist(),
    }

    # Define the features (columns)
    features = Features({
        "image": Image(),
        "text": Value("string"),
    })

    # Create the dataset
    dataset = Dataset.from_dict(data_dict, features=features)

    # Save the dataset locally
    dataset.save_to_disk(output_dir)

def upload_to_hf(hf_dataset, dataset_dir):
    dataset = load_dataset("imagefolder",
                           data_dir=dataset_dir)

    dataset.push_to_hub(hf_dataset)

if __name__ == "__main__":
    # gen_csv_to_upload("/Users/iliabenkovitch/Documents/deep_learning_raja/mocha_files/images_xl"
    #                   "/captions_5k_upload_huggingface.csv",
    #                   "/Users/iliabenkovitch/Documents/deep_learning_raja/mocha_files/images_xl"
    #                   "/captions_5k_upload_huggingface_out.csv"
    #                   )

    # gen_data_for_hf(csv_path="/Users/iliabenkovitch/Documents/deep_learning_raja/mocha_files/images_xl"
    #                       "/captions_5k_upload_huggingface_out.csv",
    #                 images_dir="/Users/iliabenkovitch/Documents/deep_learning_raja/mocha_files/images_xl/images_gpt4",
    #                 output_dir="/Users/iliabenkovitch/Documents/deep_learning_raja/mocha_files/images_xl/dataset")

    upload_to_hf(hf_dataset="iliabenk/verbs_2",
                 local_dataset="/Users/iliabenkovitch/Documents/deep_learning_raja/mocha_files/images_xl/dataset")
