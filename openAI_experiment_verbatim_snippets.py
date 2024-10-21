import pandas as pd
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
import json

# Set environment variables for LangChain API key and tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

N_WORDS_PER_CHUNK = 50000
CHUNK_STRIDE = 45000

def chunk_doc(doc, n_words_per_chunk=N_WORDS_PER_CHUNK, chunk_stride=CHUNK_STRIDE):
    # Split the document into words
    words = doc.split()
    
    # Initialize the list to store chunks
    ls_doc = []
    
    # Calculate the total number of words in the document
    total_words = len(words)
    
    # Start chunking process
    start = 0
    while start < total_words:
        # Determine the end of the current chunk
        end = start + N_WORDS_PER_CHUNK
        
        # If the end exceeds total words, adjust it
        if end > total_words:
            end = total_words
        
        # Create the current chunk and append to the list
        current_chunk = ' '.join(words[start:end])
        ls_doc.append(current_chunk)
        
        # Update the start index for the next chunk
        start += CHUNK_STRIDE
        
        # If the next chunk starts beyond the total number of words, break
        if start >= total_words:
            break
            
    return ls_doc


class SnippetDetail(BaseModel):
    """Represents extracted detail (must be full sentences in order to not lose any context) and corresponding short verbatim snippet (from the current chunk of the document)."""
    extracted_info: str = Field(
        default="Not explicitly stated",
        description="The model's understanding of the information. This must be full sentences in order to not lose any context. May indicate that no such detail is found.",
    )
    short_verbatim_snippets: List[str] = Field(
        default="",
        description="Short verbatim snippets (8-12 words each) from the original document (but the snippet has to be EXACTLY THE SAME as the snippet from the document. Do not correct any spelling errors). Can be blank if the detail cannot be found in this part.",
    )


class ComplaintSummary(BaseModel):
    """Structured extracted-key-information from the current chunk of document."""
    filing_date: SnippetDetail = Field(default=[], description="The filing date or year, if the information is clearly available.")
    court_name: SnippetDetail = Field(default=[], description="Full name of the court where the case was filed, if the information is clearly available.")
    judge_name: SnippetDetail = Field(default=[], description="The name and title of the judge, if the information is clearly available.")
    counsel_type: SnippetDetail = Field(default=[], description="The type of counsel (private, legal services, etc.), if the information is clearly available.")
    class_action_or_individuals: SnippetDetail = Field(default=[], description="Is the case a class action lawsuit or for individual plaintiffs?")
    plaintiffs_info: SnippetDetail = Field(default=[], description="Information about the plaintiffs.")
    defendants_info: SnippetDetail = Field(default=[], description="Information about the defendants and their titles.")
    legal_claims: SnippetDetail = Field(default=[], description="Legal claims and specific allegations.")
    injunctive_relief: SnippetDetail = Field(default=[], description="Injunctive relief sought and its relation to the judgment, if the information is clearly available.")
    declaratory_relief: SnippetDetail = Field(default=[], description="Declaratory relief sought, if the information is clearly available.")
    attorney_fees: SnippetDetail = Field(default=[], description="Attorney fees sought, if the information is clearly available.")
    money_damages: SnippetDetail = Field(default=[], description="Money damages sought and what kind, if the information is clearly available.")
    current_chunk_partial_summary: str = Field(default="", description="Partial summary of the current chunk, including concrete details or stories of violations.")


def generate_link_to_snippet(prefix_string, search_string, document_id, url_body="https://clearinghouse-umich-production.s3.amazonaws.com/media/doc/"):
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
    # URL encode the search string to ensure it's safely included in the URL
    import urllib.parse
    encoded_search_string = urllib.parse.quote(search_string)
    
    # Construct the full URL
    full_url = f"{prefix_string}{url_body}{document_id}.pdf#search={encoded_search_string}&phrase=true"
    
    return full_url

def print_extracted_case_info(llm_structured_output, clearinghouse_doc_id=None, generate_link_to_doc=False, prefix_string="chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/", ):
    """
    Prints the contents of a ComplaintSummary object in a human-readable format. If the `generate_link_to_doc` flag
    is True, it also generates a link for each short verbatim snippet using the generate_link_to_snippet() function.
    
    Args:
        llm_structured_output (ComplaintSummary): The complaint summary object containing extracted details from a legal case.
        generate_link_to_doc (bool, optional): Whether or not to include a clickable link next to each short verbatim snippet. 
                                               Defaults to True.
        prefix_string (str, optional): The prefix used for generating links to PDF snippets. Defaults to the PDFjs Chrome extension prefix.
        document_id (int): The document ID used to generate the links.
    """
    
    # Helper function to print the field's information
    def print_field(field_name, field_detail):
        print(f"{field_name}:")
        print(f"  Extracted Info: {field_detail.extracted_info}")
        print("  Verbatim Snippets:")
        
        for snippet in field_detail.short_verbatim_snippets:
            if generate_link_to_doc:
                assert clearinghouse_doc_id is not None
                # Generate a link to the snippet using the generate_link_to_snippet function
                link = generate_link_to_snippet(prefix_string, snippet, clearinghouse_doc_id)
                print(f"    - {snippet} (Link: {link})")
            else:
                print(f"    - {snippet}")
        print()

    # Pretty print each field
    print_field("Filing Date", llm_structured_output.filing_date)
    print_field("Court Name", llm_structured_output.court_name)
    print_field("Judge Name", llm_structured_output.judge_name)
    print_field("Counsel Type", llm_structured_output.counsel_type)
    print_field("Class Action or Individuals", llm_structured_output.class_action_or_individuals)
    print_field("Plaintiffs Info", llm_structured_output.plaintiffs_info)
    print_field("Defendants Info", llm_structured_output.defendants_info)
    print_field("Legal Claims", llm_structured_output.legal_claims)
    print_field("Injunctive Relief", llm_structured_output.injunctive_relief)
    print_field("Declaratory Relief", llm_structured_output.declaratory_relief)
    print_field("Attorney Fees", llm_structured_output.attorney_fees)
    print_field("Money Damages", llm_structured_output.money_damages)
    
    # Also print the partial summary of the current chunk
    print("Current Chunk Partial Summary:")
    print(f"  {llm_structured_output.current_chunk_partial_summary}")


def summarize_doc(document, 
                doc_name="", 
                clearinghouse_doc_id=None,
                # print_token_usage=True, 
                print_summary=True, 
                print_extracted_info=True,
                return_metadata=False,
                ):
    """Summarize 'document' passed in as a long string.
    'doc_name' is optional and is only used when printing to more easily identify which document has summary being printed.

    Return: just the summary as a string, 
            or (summary, list of all OpenAI return objects) if return_metadata is True. The OpenAI return objects are for each individual chunk and for the final combination prompt, and would help us inspect each return object for each chunk for troubleshooting if needed.
    """

    ls_doc = chunk_doc(doc)

    print(f"\nDocument {doc_name} has {len(doc.split(' '))} words and was divided into {len(ls_doc)} chunks.")
    print(f"Length of each chunk (in words): {[len(doc.split(' ')) for doc in ls_doc]}")

    # Initialize the OpenAI model
    model = ChatOpenAI(model="gpt-4o-mini")

    # Create a parser for processing model outputs
    parser = StrOutputParser()

    # Define a system message
    system_message = SystemMessage(content="You are a helpful assistant summarizing legal cases. You need to extract key information that's needed for the summarization, but you are also asked to provide short verbatim snippets from the original document, so that a human reviewer can fact-check by searching for the verbatim snippets in the original document.")

    # Define a prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_message.content), ("user", "{prompt_content}")]
    )

    structured_llm = model.with_structured_output(ComplaintSummary)

    # extracted_info will store the extracted facts that are required for each part of the original doc
    extracted_info = [] 
    return_objects = []

    # Loop through each part of the complaint
    for index, part in enumerate(ls_doc):
        # Construct the prompt content
        prompt_content = f"""
extract key fields of information that are requested from this part {index + 1} of {len(ls_doc)} of a complaint.

**Instructions:**

For each field of information, you need to extract the necessary information, along with the short verbatim snippets (there can be multiple) that are directly quoted from the document in order to support your extracted information.

Please ensure that all aspects of your extracted information are supported by the short verbatim snippets. Please ensure that your responses resemble the specified structure (the number of snippets to support each field can vary). 

The extracted information should be in full sentences to be clear and provide context, and any direct verbatim snippet from the document must be exact.

The corresponding short verbatim snippets (8-12 words, from the current chunk of the document) have to be EXACTLY THE SAME as the snippet from the document. Do not correct any spelling errors). That way, a human reviewer can fact-check by searching for the verbatim snippets in the original document. Can be blank if the detail cannot be found in this part.


**Please extract the following information:**

1. **Filing Date:** What is the date or year when the complaint was filed?
2. **Court Name:** What is the full name of the court where the case was filed?
3. **Judge Name:** What is the name and title of the judge?
4. **Counsel Type:** What type of counsel is involved (e.g., private, legal services)?
5. **Class Action or Individuals:** Is this a class action lawsuit or for individual plaintiffs?
6. **Plaintiffs Information:** Provide information about the plaintiffs involved in the case.
7. **Defendants Information:** Provide information about the defendants and their titles.
8. **Legal Claims:** What are the specific legal claims and allegations in the case?
9. **Injunctive Relief:** What injunctive relief is being sought, and how does it relate to the judgment?
10. **Declaratory Relief:** What declaratory relief is being sought, if any?
11. **Attorney Fees:** What attorney fees are being sought, if specified?
12. **Money Damages:** What money damages are being sought, if specified?
13. **Current Chunk Partial Summary:** Provide a partial summary of this section of the document, including any concrete details or stories of violations.

---

**Expected Output Structure:**

```json
{{
    "filing_date": {{
        "extracted_info": "The model's understanding of the filing date.",
        "short_verbatim_snippets": [
            "Exact snippet from the document.",
            "Additional snippet if relevant.",
            "Additional snippet if relevant.",
        ]
    }},
    "court_name": {{
        "extracted_info": "The model's understanding of the court name.",
        "short_verbatim_snippets": [
            "Exact snippet from the document.",
            "Additional snippet if relevant.",
            "Additional snippet if relevant.",
        ]
    }},
    "judge_name": {{
        "extracted_info": "The model's understanding of the judge's name.",
        "short_verbatim_snippets": [
            "Exact snippet from the document.",
            "Additional snippet if relevant.",
            "Additional snippet if relevant.",
        ]
    }},
    "counsel_type": {{
        "extracted_info": "The model's understanding of the counsel type.",
        "short_verbatim_snippets": [
            "Exact snippet from the document.",
            "Additional snippet if relevant.",
            "Additional snippet if relevant.",
        ]
    }},
    "class_action_or_individuals": {{
        "extracted_info": "The model's understanding of class action or individual status.",
        "short_verbatim_snippets": [
            "Exact snippet from the document.",
            "Additional snippet if relevant.",
            "Additional snippet if relevant.",
        ]
    }},
    "plaintiffs_info": {{
        "extracted_info": "The model's understanding of plaintiffs' information.",
        "short_verbatim_snippets": [
            "Exact snippet from the document.",
            "Additional snippet if relevant.",
            "Additional snippet if relevant.",
        ]
    }},
    "defendants_info": {{
        "extracted_info": "The model's understanding of defendants' information.",
        "short_verbatim_snippets": [
            "Exact snippet from the document.",
            "Additional snippet if relevant.",
            "Additional snippet if relevant.",
        ]
    }},
    "legal_claims": {{
        "extracted_info": "The model's understanding of legal claims.",
        "short_verbatim_snippets": [
            "Exact snippet from the document.",
            "Additional snippet if relevant.",
            "Additional snippet if relevant.",
        ]
    }},
    "injunctive_relief": {{
        "extracted_info": "The model's understanding of injunctive relief sought.",
        "short_verbatim_snippets": [
            "Exact snippet from the document.",
            "Additional snippet if relevant.",
            "Additional snippet if relevant.",
        ]
    }},
    "declaratory_relief": {{
        "extracted_info": "The model's understanding of declaratory relief sought.",
        "short_verbatim_snippets": [
            "Exact snippet from the document.",
            "Additional snippet if relevant.",
            "Additional snippet if relevant.",
        ]
    }},
    "attorney_fees": {{
        "extracted_info": "The model's understanding of attorney fees.",
        "short_verbatim_snippets": [
            "Exact snippet from the document.",
            "Additional snippet if relevant.",
            "Additional snippet if relevant.",
        ]
    }},
    "money_damages": {{
        "extracted_info": "The model's understanding of money damages sought.",
        "short_verbatim_snippets": [
            "Exact snippet from the document.",
            "Additional snippet if relevant.",
            "Additional snippet if relevant.",
        ]
    }},
    "current_chunk_partial_summary": "A summary of the current chunk."
}}
```

NOTES:

+ Snippets must be verbatim and can include spelling errors.
+ Use the provided document excerpt to base your answers.

COMPLAINT PART {index + 1} of {len(ls_doc)}
        {part}"""
        
        # Create the full prompt by substituting the template with the actual content
        full_prompt = prompt_template.invoke({"prompt_content": prompt_content})
        
        # Invoke the model with the constructed prompt
        structured_output = structured_llm.invoke(full_prompt)

        if print_extracted_info:
            print(f"EXTRACTED INFORMATION FOR DOCUMENT: {doc_name}")
            print("="*40)
            print_extracted_case_info(
                structured_output,
                clearinghouse_doc_id=clearinghouse_doc_id,
                generate_link_to_doc=True, 
                prefix_string="chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/")
            pass

        # Parse the structured model response
        string_json_output = json.dumps(json.loads(structured_output.json()), indent=2)  # Converts the Pydantic model to string

        # Save the extracted information
        extracted_info.append(string_json_output)
        # Save this return object metadata in order to compute the token usage stats at the end
        return_objects.append(structured_output)


    total_parts = len(extracted_info)

    # Construct the aggregate prompt
    aggregate_prompt = "Please provide a summary of a complaint, in about 2-4 paragraphs, based on the summaries of the individual parts:\n\n"

    for i, part_summary in enumerate(extracted_info):
        aggregate_prompt += f"NOTE: Here’s the summary of part {i + 1} of {total_parts}:\n{part_summary}\n\n"

    # Add the instructions at the end
    aggregate_prompt += """INSTRUCTION: Please include the following information in your summary (if that information is available): 
1. The filing date. If no explicit date is given but a year is, provide the year only. 
2. Full name of the court where the case was filed e.g. “U.S. District Court for the District of New York”. This should include the state district that this course is taking place in. 
3. The name and title of the Judge. Example: District Judge J. Paul Oetken. If the name of the judge is not available, please note that the judge’s not mentioned.
4. Type of counsel. Such as: private, legal services, state protection & advocacy system, ACLU, etc. Do not list organizations or names. 
5. Indicate if this is a class action lawsuit or if it involves individual plaintiffs. Do not name any attorneys as plaintiffs. 
6. Who are the defendants? Also include their titles (if mentioned) for clear contexts.
7. Who are the plaintiffs? If plaintiffs are not an organization, just describe them. 
8. The plaintiffs' legal claims, which include: The Statutory or constitutional basis for claim. If there is a state claim, note the state. In addition, mention the specific allegations (if they are mentioned).
9. As part of the remedies for the case, was injunctive relief sought? If so, describe the injunctive relief sought in relation to any judgment. 
10. As part of the remedies for the case, was Declaratory relief sought? 
11. As part of the remedies for the case, was Attorney fees sought and how much? 
12. As part of the remedies for the case, was money damages sought? If so, what kind? 
Only use the information provided in this part of the complaint to create the summary. Only provide the summary, do not generate any text outside the summary"""

    to_model_prompt = prompt_template.invoke({"prompt_content": aggregate_prompt})

    ret_val = model.invoke(to_model_prompt)
    final_summary = ret_val.content
    return_objects.append(ret_val)

    # if print_token_usage:
        # pass
        # # Compute the stats
        # total_input_tokens = 0
        # total_output_tokens = 0

        # # Loop through each return object
        # for obj in return_objects:
        #     total_input_tokens += obj.usage_metadata['input_tokens']
        #     total_output_tokens += obj.usage_metadata['output_tokens']

        # print(f"\nTotal Input Tokens: {total_input_tokens}")
        # print(f"Total Output Tokens: {total_output_tokens}")

    if print_summary:
        print(f"\n\nSUMMARY {doc_name}\n{final_summary}")

    if return_metadata:
        return final_summary, return_objects
    
    return final_summary

# UPDATE this path to point to where your .csv dataset is. I'm using this version: https://3.basecamp.com/5835116/buckets/38747617/messages/7891399906
# ALSO set the list of indices among the 40+ complaints in this .csv file

# courtesy of Sam Pang's test set: https://3.basecamp.com/5835116/buckets/38747617/messages/7916029720
document_IDs = [16, 37, 24, 40, 38, 0, 22, 13, 33, 34, 1]
clearinghouse_doc_ids = [105652, 139419, 133296, 145814, 139431, 612, 130478, 101295, 135664, 135665, 613]
case_names = ['People ex rel. Harpaz o/b/o Vance v. Brann',
 'Adkins v. State of Idaho',
 'Munday v. Beaufort County',
 'Planned Parenthood South Atlantic v. South Carolina',
 'Planned Parenthood South Atlantic v. State of South Carolina',
 'Macer v. Dinisio',
 'Kariye v. Mayorkas',
 'T.M. v. City of Philadelphia',
 'Sanders v. District of Columbia',
 'Strifling v. Twitter, Inc.',
 "Mohler v. Prince George's County"]

df = pd.read_csv("/home/hice1/tnguyen868/scratch/openai-summarization/parsed_documents.csv", sep="|")

for i, complaint_idx in enumerate(document_IDs):
    doc = df["Document"].iloc[complaint_idx]
    summary, metadata = summarize_doc(
        doc, 
        doc_name=f"COMPLAINT #{complaint_idx} - \"{case_names[i]}\"",
        clearinghouse_doc_id=clearinghouse_doc_ids[i],
        print_summary=True,
        print_extracted_info=True,
        return_metadata=True
    )
    pass 

