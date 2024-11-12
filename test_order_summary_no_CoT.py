import os
import pandas as pd
from langchain_openai import ChatOpenAI
from legal_doc_summarizer import DocumentSummarizer, extract_doc_id

# Initialize the LLM instance
llm = ChatOpenAI(model="gpt-4o-mini")

# Load the CSV file
file_path = '/home/hice1/tnguyen868/scratch/openai-summarization/clearinghouse_order_ops_ocr_df_complete.csv'
df = pd.read_csv(file_path)

# Define the indices to loop through
document_indices = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]

# Define the questions list (constant for all documents)
question_list = [
    # Outline of Ruling
    "1. What is the main issue or issues the judge is ruling on in this document?",
    "2. Does the introductory paragraph offer a summary of the ruling, and if so, what key points or decisions does it outline?",
    "3. What is the overall ruling (e.g., is a Preliminary Injunction granted or denied, or granted in part and denied in part)?",
    
    # Discussion Section
    "4. Does the judge discuss 'standing' as part of the case, and if so, what are the primary points or concerns related to standing?",
    "5. Is standing in contention, and what arguments does the court consider to determine it?",
    "6. What are the main arguments presented by the defendant, and how does the court respond to them?",
    "7. What does the court conclude regarding the 'likelihood of success on the merits' for the claims made?",
    
    # Relief Granted
    "8. What specific relief does the court grant or deny (e.g., summary judgment, injunction, or motion to dismiss)?",
    "9. Are there any particular conditions or limitations placed on the relief granted?",
    "10. Does the document specify if relief is partial (e.g., injunction is granted in part and denied in part), and if so, what distinctions does the judge make?",
    
    # Conclusion
    "11. What judgments or orders does the court list in the conclusion or final section?",
    "12. Are there any other significant points or directives provided in the conclusion that indicate next steps or implications of the ruling?"
]

# Initialize the summarizer (constant for all documents)
summarizer = DocumentSummarizer(llm, question_list)

# Loop through the specified indices
for index in document_indices:
    # Extract the row data for the current index
    row = df.iloc[index]
    input_text = row.get('clean_text', '')

    # If input text is missing, skip to the next document
    if not input_text:
        print(f"Skipping index {index} due to missing input text.")
        continue

    # Extract metadata from the current row
    metadata = {
        "file_name": row.get("file_name", ""),
        "case_id": row.get("case_id", ""),
        "case_name": row.get("case_name", ""),
        "summary": row.get("summary", ""),
        "court": row.get("court", ""),
        "filing_date": row.get("filing_date", ""),
        "doc_url": row.get("doc_url", "")
    }

    print(f"***** PROCESSING DOCUMENT {extract_doc_id(row.get('doc_url', ''))} ******")

    # Define the output file path
    output_file = f"/home/hice1/tnguyen868/scratch/openai-summarization/court_order_summaries_ver4/{extract_doc_id(row.get('doc_url', ''))}.txt"

    try:
        # Generate the summary response
        summary_response = summarizer.summarize_document(input_text)

        # Save the output in both JSON and human-readable formats
        summarizer.save_case_summary(
            metadata=metadata,
            summary_response=summary_response,
            save_as_json=False,
            output_file=output_file,
            print_to_screen=True,
            generate_link=True
        )
    except Exception as e:
        print(f"Error processing document at index {index}: {str(e)}")
        continue

print("Processing completed.")
