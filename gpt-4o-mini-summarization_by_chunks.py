import pandas as pd
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Set environment variables for LangChain API key and tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langchain-api-key"
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

N_WORDS_PER_CHUNK = 2000
CHUNK_STRIDE = 1500

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



def summarize_doc(document, 
                doc_name="", 
                print_token_usage=True, 
                print_summary=True, 
                return_metadata=False):
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
    system_message = SystemMessage(content="You are a helpful assistant summarizing legal cases.")

    # Define a prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_message.content), ("user", "{prompt_content}")]
    )

    # extracted_info will store the extracted facts that are required for each part of the original doc
    extracted_info = [] 
    return_objects = []

    # Loop through each part of the complaint
    for index, part in enumerate(ls_doc):
        # Construct the prompt content
        prompt_content = f"""
summarize this part {index + 1} of {len(ls_doc)} of a complaint.

INSTRUCTION: Please include the following information in your summary (if that information is available): 
1. The filing date. If no explicit date is given but a year is, provide the year only. If no filing date is detected, please note that fact.
2. Full name of the court where the case was filed e.g. “U.S. District Court for the District of New York”.
3. The name and title of the Judge. If not available, please note that.
4. Type of counsel. Such as: private, legal services, etc. 
5. Class action lawsuit or individual plaintiffs. Do not name any attorneys as plaintiffs. 
6. The defendants and their titles for context.
7. The plaintiffs, describe them if not an organization. 
8. Legal claims including statutory or constitutional basis. Note specific allegations.
9. Injunctive relief sought and its relation to judgement.
10. Declaratory relief sought.
11. Attorney fees sought and how much.
12. Money damages sought and what kind.
Only use the information provided in this part of the complaint to create the summary.

COMPLAINT PART {index + 1} of {len(ls_doc)}
        {part}
        """
        
        # Create the full prompt by substituting the template with the actual content
        full_prompt = prompt_template.invoke({"prompt_content": prompt_content})
        
        # Invoke the model with the constructed prompt
        ret_val = model.invoke(full_prompt)


        # Parse the model response
        parsed_output = parser.invoke(ret_val)

        # Save the extracted information
        extracted_info.append(parsed_output)
        # Save this return object metadata in order to compute the token usage stats at the end
        return_objects.append(ret_val)


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

    if print_token_usage:
        # Compute the stats
        total_input_tokens = 0
        total_output_tokens = 0

        # Loop through each return object
        for obj in return_objects:
            total_input_tokens += obj.usage_metadata['input_tokens']
            total_output_tokens += obj.usage_metadata['output_tokens']

        print(f"\nTotal Input Tokens: {total_input_tokens}")
        print(f"Total Output Tokens: {total_output_tokens}")

    if print_summary:
        print(f"\n\nSUMMARY {doc_name}\n{final_summary}")

    if return_metadata:
        return final_summary, return_objects
    
    return final_summary

# UPDATE this path to point to where your .csv dataset is. I'm using this version: https://3.basecamp.com/5835116/buckets/38747617/messages/7891399906
# ALSO set the list of indices among the 40+ complaints in this .csv file

# courtesy of Sam Pang's test set: https://3.basecamp.com/5835116/buckets/38747617/messages/7916029720
document_IDs = [16, 37, 24, 40, 38, 0, 22, 13, 33, 34, 1]
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
    summary, metadata = summarize_doc(doc, doc_name=f"COMPLAINT #{complaint_idx} - \"{case_names[i]}\"", print_summary=True, print_token_usage=True, return_metadata=True)
    pass 

