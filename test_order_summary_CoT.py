import pandas as pd
import re
from legal_doc_summarizer_Chain_of_Thought import Summarizer
from langchain_openai import ChatOpenAI

# Define the list of document indices to process
document_indices = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]

# Load the CSV file
file_path = '/home/hice1/tnguyen868/scratch/openai-summarization/clearinghouse_order_ops_ocr_df_complete.csv'
df = pd.read_csv(file_path)

# Function to extract the document ID from the URL
def extract_doc_id(url):
    match = re.search(r'/(\d+)\.pdf$', url)
    return int(match.group(1)) if match else None

# Initialize the LLM (adjust model if needed)
llm = ChatOpenAI(model="gpt-4o-mini")

# Loop through the specified document indices
for index in document_indices:
    # Extract the relevant row based on the index
    row = df.iloc[index]
    
    # Extract document details
    doc_url = row['doc_url']
    document_id = extract_doc_id(doc_url)
    file_name = row['file_name']
    input_text = row['clean_text']
    
    if document_id is None:
        print(f"Could not extract document ID for index {index}")
        continue

    print(f"\n\n**** Processing document ID: {document_id} (Index in dataframe: {index})")
    print(f"**** Case {row['case_name']} - Document {row['file_name']} ****")

    # Define the list of questions for the summarizer
    
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


    # Initialize the Summarizer
    summarizer = Summarizer(
        llm=llm,
        input_text=input_text,
        clearinghouse_doc_id=document_id,
        question_list=question_list,
        document_type="order/opinion",
        short_sentence_length=11,
        num_short_sentences=20,
        quote_length=10,
    )

    # Extract sentences for the questions
    extracted_info_all_chunks = summarizer.extract_sentences_for_questions()

    # Generate output files using the extracted document ID
    extracted_info_file = f"/home/hice1/tnguyen868/scratch/openai-summarization/order_opinion_texts/{document_id}_extracted_info.txt"
    summarizer.print_extracted_info(
        extracted_info_all_chunks,
        generate_link=True,
        save_to_file=True,
        print_to_screen=False,
        output_file=extracted_info_file
    )

    # Generate a concise summary
    summary = summarizer.generate_concise_summary(
        extracted_info_all_chunks,
        question_list=question_list,
        save_to_file=True,
        print_to_screen=True,
        output_file=f"/home/hice1/tnguyen868/scratch/openai-summarization/order_opinion_texts/{document_id}_summary.txt"
    )

    # Identify supporting quotes for the summary
    verification_results = summarizer.identify_quotes_for_summary(
        summary=summary,
        extracted_info_all_chunks=extracted_info_all_chunks,
        save_to_file=True,
        print_to_screen=False,
        output_file=f"/home/hice1/tnguyen868/scratch/openai-summarization/order_opinion_texts/{document_id}_summary_quotes.txt",
        output_format='yaml',
    )

    print(f"Completed processing for document ID: {document_id}\n")

print("Processing complete.")
