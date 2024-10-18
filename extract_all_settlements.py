import pandas as pd
from datasets import load_dataset
import csv  # Importing csv module for saving the DataFrame

# Load the MultiLex-Sum dataset
cache_directory = "/home/hice1/tnguyen868/scratch/multilex-sum-demo/cache_dir"
multi_lexsum = load_dataset("allenai/multi_lexsum", name="v20230518", cache_dir=cache_directory)

# Initialize a list to store rows of the dataframe
rows = []

# Function to process each dataset split (train, validation, test)
def process_split(split_name, dataset_split):
    """
    Loops through each example in the dataset split, checks for doc_type 'Settlement Agreement',
    and compiles relevant information into a dataframe row.
    """
    for example in dataset_split:  # Loop through each example in the split
        case_name = example["case_metadata"]["case_name"]
        case_filing_date = example["case_metadata"]["filing_date"]

        # Get sources and their corresponding metadata
        sources = example["sources"]
        sources_metadata = example["sources_metadata"]
        
        # Get the long summary of the case
        long_summary = example["summary/long"]  # Extracting the long summary

        # Loop through each document in sources_metadata to check for 'Settlement Agreement'
        for i, doc_type in enumerate(sources_metadata["doc_type"]):
            if doc_type == "Settlement Agreement":  # Check if the doc_type is 'Settlement Agreement'
                # Compile the relevant fields into a dictionary (row for the dataframe)
                row = {
                    "doc_id": sources_metadata["doc_id"][i],  # Document ID
                    "example_id": example["id"],  # Example ID
                    "which_split": split_name,  # Split (train, validation, test)
                    "doc_type": doc_type,  # Document type ('Settlement Agreement')
                    "doc_title": sources_metadata["doc_title"][i],  # Document title
                    "doc_url": sources_metadata["url"][i],  # Document URL
                    "source_text": sources[i],  # Full text of the source document
                    "source_length_words": len(sources[i].split()),  # Length of source text in words
                    "case_name": case_name,  # Case name from case_metadata
                    "case_filing_date": case_filing_date,  # Filing date from case_metadata
                    "long_summary": long_summary  # Include the long summary
                }
                
                # Append the row to the list
                rows.append(row)

# Loop through each split in the dataset (train, validation, test)
for split_name, dataset_split in multi_lexsum.items():
    print(f"Processing {split_name} split with {len(dataset_split)} examples.")
    process_split(split_name, dataset_split)  # Process each split

# Convert the collected rows into a DataFrame
df = pd.DataFrame(rows)

# Save the DataFrame to CSV in the current directory
df.to_csv(
    "settlements_multi_lexsum.csv",
    index=False,
    escapechar='\\',  # Use backslash as escape character
    quoting=csv.QUOTE_MINIMAL  # Quote fields with special characters
)

# Display the first few rows of the dataframe to inspect
print(df.head())
