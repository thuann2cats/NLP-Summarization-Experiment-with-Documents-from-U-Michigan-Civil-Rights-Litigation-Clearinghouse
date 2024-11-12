import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
import pandas as pd
import re

# Set environment variables for LangChain API key and tracing
# these two environment variables must be set:
# OPENAI_API_KEY
# LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# For PDFjs extension on Chrome. Update this to reflect the prefix of your PDFjs instance on Chrome
PDFJS_PREFIX_STRING = "chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/"

class KeyPhraseQuote(BaseModel):
    key_phrase: str
    quote: str

class SentenceAnalysis(BaseModel):
    summary_sentence: str
    key_phrases_with_quotes: List[KeyPhraseQuote]

class SummaryResponse(BaseModel):
    summary: str
    quotes: List[SentenceAnalysis]


def generate_link_to_snippet(
    search_string: str = None, 
    document_id: int = None, 
    prefix_string=PDFJS_PREFIX_STRING, 
    url_body="https://clearinghouse-umich-production.s3.amazonaws.com/media/doc/"):
    """
    Generates a URL that can be used with the PDFjs extension on Google Chrome to render a PDF 
    and highlight a specific search string.

    This function builds a URL by combining a Chrome extension prefix, a base URL for the document,
    and additional query parameters for highlighting the search term in the PDF viewer. The PDF 
    will load and automatically highlight the search term if found.

    Args:
        prefix_string (str): The URL prefix used by the PDFjs extension, typically starting with 
                            "chrome-extension://", e.g., "chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/".
        search_string (str): The phrase to be highlighted in the PDF. This will be URL-encoded 
                            to ensure it's formatted correctly for the query string.
        document_id (int): The unique identifier for the document, typically used as part of the 
                        filename in the base URL. This number will be inserted into the URL 
                        before ".pdf".
        url_body (str, optional): The base URL where the documents are hosted. Defaults to 
                                "https://clearinghouse-umich-production.s3.amazonaws.com/media/doc/".

    Returns:
        str: A fully formed URL string that can be pasted into Google Chrome and used with the 
            PDFjs extension to render the PDF and highlight the search term.
    
    Example:
        >>> generate_link_to_snippet(
                "chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/", 
                "respectfully submitted", 
                34698
            )
        'chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://clearinghouse-umich-production.s3.amazonaws.com/media/doc/34698.pdf#search=respectfully%20submitted&phrase=true'
    """
    if search_string is None or document_id is None:
        return "<ERROR GENERATING LINK>"

    # URL encode the search string to ensure it's safely included in the URL
    import urllib.parse
    encoded_search_string = urllib.parse.quote(search_string)
    
    # Construct the full URL
    full_url = f"{prefix_string}{url_body}{document_id}.pdf#search={encoded_search_string}&phrase=true"
    
    return full_url

# Function to extract the document ID from the URL
def extract_doc_id(url):
    match = re.search(r'/(\d+)\.pdf$', url)
    return int(match.group(1)) if match else None


class DocumentSummarizer:
    def __init__(self, llm, question_list: List[str]):
        self.llm = llm
        self.question_list = question_list

    def summarize_document(self, document: str) -> SummaryResponse:
        system_message = """
You are a helpful assistant for summarizing legal documents. Extract key points for summarization and provide verbatim quotes supporting key phrases.
"""

        user_message_template = """
summarize the following document into a summary that is about 2-3 paragraphs.

be sure to address these questions (if relevant):
{question_list}

Then break down the summary into individual sentences. For each sentence, identify key phrases and quote relevant short verbatim snippets from the document that support the key phrases.

** INSTRUCTIONS **

Generate output in the following JSON format:

```json
{{
    "summary": "The full summary text here.",
    "quotes": [
        {{
            "summary_sentence": "First sentence from the summary.",
            "key_phrases_with_quotes": [
                {{
                    "key_phrase": "A key phrase in the sentence.",
                    "quote": "A direct quote from the document supporting this key phrase (no more than 6 words, do not put any punctuations before and after the quote)."
                }},
                {{
                    "key_phrase": "Another key phrase in the sentence.",
                    "quote": "Another direct quote from the document supporting this key phrase (no more than 6 words, do not put any punctuations before and after the quote)."
                }},
                ... more pairs of key phrase and quote, as necessary. Generate at least 3.
            ]
        }},
        {{
            "summary_sentence": "Second sentence from the summary.",
            "key_phrases_with_quotes": [
                {{
                    "key_phrase": "A key phrase in the sentence.",
                    "quote": "A direct quote from the document supporting this key phrase (no more than 6 words, do not put any punctuations before and after the quote)."
                }},
                ... more pairs of key phrase and quote, as necessary. Generate at least 3.
            ]
        }}
        ... (more sentences with key phrases and quotes)
    ]
}}
```

** DOCUMENT TO BE SUMMARIZED **
{document}
"""

        # Set up the prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(
                user_message_template,
                input_variables=["document", "question_list"]
            )
        ])

        # Define structured output for the response
        structured_output = self.llm.with_structured_output(SummaryResponse)
        chain = prompt_template | structured_output

        response = chain.invoke({
            "document": document,
            "question_list": "\n".join(self.question_list)
        })
        return response

    def save_case_summary(
        self,
        metadata: Dict[str, Any],
        summary_response: SummaryResponse,
        save_as_json: bool = True,
        output_file: str = None,
        print_to_screen: bool = False,
        generate_link: bool = True,
    ):
        """
        Save the case summary along with metadata either as a JSON file or in a human-readable format.
        """
        # Create the combined dictionary
        case_dict = {
            "case_metadata": metadata,
            "case_summary": summary_response.dict()
        }

        if save_as_json:
            # Save in JSON format
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(case_dict, f, indent=4)
                    print(f"Case summary saved to {output_file} in JSON format.")

            # Print to screen if flag is set
            if print_to_screen:
                print("\n# CASE SUMMARY (JSON FORMAT):")
                print(json.dumps(case_dict, indent=4))
        
        else:
            # Save in human-readable format
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(self._format_human_readable(metadata, summary_response))
                    print(f"Case summary saved to {output_file} in human-readable format.")
            
            # Print to screen if flag is set
            if print_to_screen:
                print("\n# CASE SUMMARY (Human-Readable Format):")
                print(self._format_human_readable(metadata, summary_response, generate_link=generate_link))

    def _format_human_readable(self, metadata: Dict[str, Any], summary_response: SummaryResponse, generate_link=True) -> str:
        """
        Format the metadata and summary response in a human-readable format.
        """
        formatted_output = []

        # Add metadata
        import pprint
        formatted_output.append("# CASE METADATA:\n")
        for key, value in metadata.items():
            formatted_output.append(f"{key}: {value}")

        # Add summary
        formatted_output.append("\n# AI-GENERATED SUMMARY:\n")
        formatted_output.append("## Summary:\n")
        formatted_output.append(summary_response.summary + "\n")

        # Add detailed analysis
        formatted_output.append("\n## Relevant Quotes for Each Sentence:\n")
        for analysis in summary_response.quotes:
            formatted_output.append(f"\nSentence: {analysis.summary_sentence}")
            for phrase in analysis.key_phrases_with_quotes:
                formatted_output.append(f"  Key Phrase: {phrase.key_phrase}")
                formatted_output.append(f"  Quote: {phrase.quote}")
                if generate_link:
                    formatted_output.append(f"  Link: {generate_link_to_snippet(phrase.quote, extract_doc_id(metadata['doc_url']))}")

        return "\n".join(formatted_output)


# llm = ChatOpenAI(model="gpt-4o-mini") # Initialize your LLM instance 

# # Load the CSV file
# file_path = '/home/hice1/tnguyen868/scratch/openai-summarization/clearinghouse_order_ops_ocr_df_complete.csv'
# df = pd.read_csv(file_path)



# index = 10
# row = df.iloc[index]
# input_text = row['clean_text']

# # output path
# output_file = f"/home/hice1/tnguyen868/scratch/openai-summarization/court_order_summaries_ver4/{extract_doc_id(row.get('doc_url', ''))}.txt"

# question_list = [
#     # Outline of Ruling
#     "1. What is the main issue or issues the judge is ruling on in this document?",
#     "2. Does the introductory paragraph offer a summary of the ruling, and if so, what key points or decisions does it outline?",
#     "3. What is the overall ruling (e.g., is a Preliminary Injunction granted or denied, or granted in part and denied in part)?",
    
#     # Discussion Section
#     "4. Does the judge discuss 'standing' as part of the case, and if so, what are the primary points or concerns related to standing?",
#     "5. Is standing in contention, and what arguments does the court consider to determine it?",
#     "6. What are the main arguments presented by the defendant, and how does the court respond to them?",
#     "7. What does the court conclude regarding the 'likelihood of success on the merits' for the claims made?",
    
#     # Relief Granted
#     "8. What specific relief does the court grant or deny (e.g., summary judgment, injunction, or motion to dismiss)?",
#     "9. Are there any particular conditions or limitations placed on the relief granted?",
#     "10. Does the document specify if relief is partial (e.g., injunction is granted in part and denied in part), and if so, what distinctions does the judge make?",
    
#     # Conclusion
#     "11. What judgments or orders does the court list in the conclusion or final section?",
#     "12. Are there any other significant points or directives provided in the conclusion that indicate next steps or implications of the ruling?"
# ]


# metadata = {
#             "file_name": row.get("file_name", ""),
#             "case_id": row.get("case_id", ""),
#             "case_name": row.get("case_name", ""),
#             "summary": row.get("summary", ""),
#             "court": row.get("court", ""),
#             "filing_date": row.get("filing_date", ""),
#             "doc_url": row.get("doc_url", "")
#         }

# summarizer = DocumentSummarizer(llm, question_list) 
# summary_response = summarizer.summarize_document(input_text)

# # Save the output in both JSON and human-readable format
# summarizer.save_case_summary( 
#     metadata=metadata, 
#     summary_response=summary_response, 
#     save_as_json=False, 
#     output_file=output_file, 
#     print_to_screen=True,
#     generate_link=True )

# pass
