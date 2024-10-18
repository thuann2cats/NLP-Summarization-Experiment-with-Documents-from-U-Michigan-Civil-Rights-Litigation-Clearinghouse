import pandas as pd
import csv

# Load the DataFrame from the CSV file in the cache directory
cache_directory = "/home/hice1/tnguyen868/scratch/multilex-sum-demo"
csv_file_path = f"{cache_directory}/settlements_multi_lexsum.csv"  # Ensure the path is correct

# Read the CSV file into a DataFrame
df_loaded = pd.read_csv(
    csv_file_path,
    escapechar='\\',  # Ensure the escape character is consistent with how it was saved
    quoting=csv.QUOTE_MINIMAL  # Same quoting style as when saving
)

# Specify the document IDs to select
selected_doc_ids = [
    'DR-CO-0020-0022', 'EE-TX-0369-0003', 'JC-CA-0042-0008', 'EE-NJ-0102-0002',
    'EE-CO-0015-0007', 'EE-MD-0105-0004', 'PN-AZ-0004-0007', 'FH-NY-0017-0002',
    'EE-WA-0108-0004', 'EE-TX-0396-0003', 'EE-NJ-0040-0002', 'EE-UT-0010-0002',
    'EE-TX-0158-0002', 'PC-AZ-0021-0005', 'JC-CA-0050-0002', 'CJ-DC-0002-0003',
    'EE-CA-0092-0003', 'EE-IL-0227-0002', 'EE-TN-0145-0002', 'JI-OH-0006-0029'
]

# Create a test set DataFrame by filtering the loaded DataFrame
test_set_df = df_loaded[df_loaded['doc_id'].isin(selected_doc_ids)]

# Display the first 500 characters for the first several documents along with the specified metadata
for index, row in test_set_df.head(3).iterrows():
    print(f"Document {index + 1}:")
    print(f"doc_id: {row['doc_id']}")
    print(f"example_id: {row['example_id']}")
    print(f"which_split: {row['which_split']}")
    print(f"case_name: {row['case_name']}")
    print(f"doc_type: {row['doc_type']}")
    print(f"doc_title: {row['doc_title']}")
    print(f"doc_url: {row['doc_url']}")
    print(f"case_filing_date: {row['case_filing_date']}")
    print(f"source_length_words: {row['source_length_words']}")
    print(f"source_text (first 500 chars): {row['source_text'][:500]}")
    print(f"long_summary (first 250 chars): {row['long_summary'][:250]}")
    print("\n" + "="*40 + "\n")  # Separator for clarity
