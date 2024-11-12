from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
# from langchain.runnables import RunnableSequence
import os
from typing import List
from pydantic import BaseModel, Field

# Set environment variables for LangChain API key and tracing
# these two environment variables must be set:
# OPENAI_API_KEY
# LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"


N_WORDS_PER_CHUNK = 50000
CHUNK_STRIDE = 45000


class ShortSentence(BaseModel):
    """
    Represents a short, single-clause sentence (fewer than 11 words) summarizing a key idea from the original
    document, paired with a short direct quote that supports the summary.
    """
    short_sentence: str = Field(
        "",
        description="A single-clause sentence under 11 words representing a single important detail, idea, or piece of information from the document."
    )
    short_verbatim_snippet: str = Field(
        "",
        description="A direct quote (under 10 words) from the original document that supports the current summary sentence, quoted verbatim without correcting any spelling errors."
    )


# class ShortSentenceList(BaseModel):
#     """
#     A list of ShortSentence objects, each containing a short sentence and verbatim snippet summarizing a single
#     important aspect of the document. The list captures the main points of the document, and some sentences may 
#     overlap in details to ensure comprehensive coverage of the document's key developments.
#     """
#     sentences: List[ShortSentence] = Field(
#         default=[],
#         description="List of ShortSentence objects capturing the most important information from the original document, with potential overlap to cover the whole story or developments."
#     )


# Pydantic Classes for Question and Answer Structure
class QuestionAndAnswers(BaseModel):
    """
    Represents a question and the associated list of ShortSentence objects that answer it.
    """
    question: str = Field(
        ...,
        description="A specific question related to the legal document."
    )
    answers: List[ShortSentence] = Field(
        default=[],
        description="List of ShortSentence objects capturing the most important information from the original document, with potential overlap to cover the whole story or developments."
    )


class QuestionAndAnswerList(BaseModel):
    """
    A list of QuestionAndAnswers objects, where each object contains a question and associated ShortSentenceList.
    """
    response: List[QuestionAndAnswers] = Field(
        default=[],
        description="List of QuestionAndAnswers objects, each containing a question and relevant ShortSentenceList answers."
    )


class SentenceVerification(BaseModel):
    """
    Represents the mapping between a sentence in the summary and its supporting short sentences.
    """
    summary_sentence: str = Field(
        ...,
        description="A sentence from the generated summary."
    )
    supporting_sentences: List[ShortSentence] = Field(
        default=[],
        description="List of ShortSentence objects that support the given summary sentence."
    )


class SentenceVerificationList(BaseModel):
    """
    Represents a list of SentenceVerification objects to validate the output.
    """
    verification: List[SentenceVerification] = Field(
        default=[],
        description="List of SentenceVerification objects, each containing a summary sentence and its supporting short sentences."
    )



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



class Summarizer:
    def __init__(self, 
        llm, 
        input_text,
        clearinghouse_doc_id: int = None, 
        prefix_string: str = "chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/",
        question_list: List = None,
        document_type: str = "order/opinion", 
        short_sentence_length: int = 11, 
        num_short_sentences: int = 20, 
        quote_length: int = 10,
    ):
        self.llm = llm
        self.input_text = input_text
        self.clearinghouse_doc_id = clearinghouse_doc_id
        self.prefix_string = prefix_string
        self.question_list = question_list
        self.document_type = document_type
        self.short_sentence_length = short_sentence_length
        self.num_short_sentences = num_short_sentences
        self.quote_length = quote_length

    def extract_sentences_for_questions(self, 
        question_list: List[str] = None, 
    ):

        if not self.question_list:
            assert question_list is not None and len(question_list) > 0
        if question_list is None or len(question_list) == 0:
            question_list = self.question_list

# Define the system and user messages
        system_message = """
You are a helpful assistant summarizing documents from legal cases. You need to extract key information that's needed for the summarization, but you are also asked to provide short verbatim snippets from the original document, so that a human reviewer can fact-check by searching for the verbatim snippets in the original document.
"""

        user_message = """
** TASK **
extract key fields of information that are requested from this part {part_index} of {num_all_parts} of a legal document that is {document_type}.

** INSTRUCTIONS **
You are given a list of questions. Please answer the following questions based on this {document_type} document chunk by giving a list of short summary sentences (no more than {short_sentence_length} words) for each question. Each answer list (for each quesiton) should cover all important aspects and details for that question and should contain no fewer than {num_short_sentences} short sentences (for each question) in order to cover all important aspects and details. The sentences might have overlapping details if necessary.

IMPORTANT: There must be each pair of question/answers for each of the given questions. If the question is not relevant, please still include the question in your generated response and associate it with an empty list.

The sentences need to be short (but complete) sentences with one clause only, instead of complicated multi-clause sentences. For each sentence, generate a short verbatim snippet (no more than {quote_length} words) from the original document that directly supports that summary sentence. For the verbatim snippet, quote verbatim from the original document and do not correct any spelling errors.

Please only use the information from the provided document chunk.

** QUESTIONS TO ANSWER (treat each element of this list as a separate question) **
There are {num_questions} questions.
{question_list}

** EXPECTED OUTPUT STRUCTURE **
Please generate a valid JSON object in the exact format provided below.
Each entry in the "response" array must be a dictionary with a "question" field (string) and an "answers" field (list of dictionaries).
Do not include any extra characters or stray text. Ensure all entries are valid dictionaries.

```json
{{
    "response": [
        {{
            "question": "The first question being answered here.",
            "answers": [
                {{
                    "short_sentence": "A single-clause sentence no more than {short_sentence_length} words representing a single important detail, idea, or piece of information to answer the corresponding question.",
                    "short_verbatim_snippet": "A direct quote (no more than {quote_length} words) from the original document that supports the current summary sentence, quoted verbatim without correcting any spelling errors. Please do not add any period at the end of this string."
                }},
                {{
                    "short_sentence": "Another single-clause sentence with important information to answer the corresponding question.",
                    "short_verbatim_snippet": "Another direct quote from the document that supports the sentence. Please do not add any period at the end of this string."
                }},
                ... more sentences as necessary (no fewer than {num_short_sentences} sentences)
            ]
        }},
        {{
            "question": "The second question being answered here.",
            "answers": [
                {{
                    "short_sentence": "Another single-clause sentence with important information to answer the corresponding question.",
                    "short_verbatim_snippet": "Another direct quote from the document that supports the sentence. Please do not add any period at the end of this string."
                }},
                ... more sentences as necessary (no fewer than {num_short_sentences} sentences)
            ]
        }},
        ... more questions and answers as necessary. There must be each pair of question/answers for each of the given questions. If the question is not relevant, please still include the question in your generated response and associate it with an empty list. Please ensure that there are only {num_questions} question-and-answer pairs here.
    ]
}}

** DOCUMENT CHUNK {part_index} of {num_all_parts}**

{input_text} """

        #Create the prompt template with input variables
        prompt = ChatPromptTemplate.from_messages( 
            messages=[ 
                SystemMessagePromptTemplate.from_template(system_message), 
                HumanMessagePromptTemplate.from_template(user_message, input_variables=["part_index", "num_all_parts", "document_type", "short_sentence_length", "num_short_sentences", "quote_length", "input_text", "question_list", "num_questions"]) 
            ]
        )

        # Wrap the LLM for structured generation with the expected output model
        llm_with_structure = self.llm.with_structured_output(QuestionAndAnswerList)

        # Create a runnable sequence that combines the prompt template and the LLM
        chain = prompt | llm_with_structure

        # chunk the text
        chunks = chunk_doc(self.input_text)

        extracted_info_all_chunks = []

        for part_index, chunk in enumerate(chunks):

            # Define the input variables
            input_variables = {
                "part_index": part_index + 1, 
                "num_all_parts": len(chunks),
                "document_type": self.document_type,
                "short_sentence_length": self.short_sentence_length,
                "num_short_sentences": self.num_short_sentences,
                "quote_length": self.quote_length,
                "input_text": chunk,
                "question_list": question_list,
                "num_questions": len(question_list)
            }

            # Invoke the chain with the input variables
            output = chain.invoke(input_variables)
            extracted_info_all_chunks.append(output)

        return extracted_info_all_chunks
    
    def generate_link_to_snippet(self, prefix_string, search_string, document_id, url_body="https://clearinghouse-umich-production.s3.amazonaws.com/media/doc/"):
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

    def print_extracted_info(self, 
        extracted_info_all_chunks, 
        generate_link: bool = False, 
        print_to_screen: bool = False,
        save_to_file: bool = True,
        output_file: str = None):

        json_output_all_chunks = []

        for llm_structured_output in extracted_info_all_chunks:
            # Create output structure with optional links
            output_data = {"response": []}
            for qa_pair in llm_structured_output.response:
                question_data = {
                    "question": qa_pair.question,
                    "answers": []
                }
                
                for short_sentence in qa_pair.answers:
                    answer_data = {
                        "short_sentence": short_sentence.short_sentence,
                        "short_verbatim_snippet": short_sentence.short_verbatim_snippet
                    }
                    
                    # Add link if generate_link is True
                    if generate_link and self.clearinghouse_doc_id:
                        answer_data["direct_link"] = self.generate_link_to_snippet(
                            self.prefix_string, short_sentence.short_verbatim_snippet, self.clearinghouse_doc_id
                        )
                    
                    question_data["answers"].append(answer_data)
                
                output_data["response"].append(question_data)

            json_output_all_chunks.append(output_data)

        # Output the data to the specified file or pretty print
        if save_to_file:
            assert output_file is not None
            import json
            with open(output_file, 'w') as f:
                json.dump(json_output_all_chunks, f, indent=4)
                print(f"Extracted info successfully written to {output_file}")
        if print_to_screen:
            import pprint
            for i, output_data in enumerate(json_output_all_chunks):
                print(f"\n----- CHUNK #{i+1} ---------------------------------")
                print(f"----------------------------------------------------")
                pprint.pprint(output_data, indent=2)

    def generate_concise_summary(self, 
        extracted_info_all_chunks, 
        question_list = None, 
        save_to_file: bool = True,
        print_to_screen: bool = False,
        output_file: str = None):

        if not self.question_list:
            assert question_list is not None and len(question_list) > 0
        if question_list is None or len(question_list) == 0:
            question_list = self.question_list
            
        # Format the extracted info into text format for the prompt
        formatted_info_all_chunks = ""

        for chunk_index, extracted_info in enumerate(extracted_info_all_chunks):
            formatted_info = ""
            for qa in extracted_info.response:
                formatted_info += f"\nQuestion: {qa.question}\n"
                for detail in qa.answers:
                    formatted_info += f"- {detail.short_sentence} (verbatim snippet: '{detail.short_verbatim_snippet}')\n"
            formatted_info_all_chunks = f"\n** CHUNK {chunk_index + 1} out of {len(extracted_info_all_chunks)} chunks:** \n"
            formatted_info_all_chunks += formatted_info
        
        # System and user messages for summarizing the extracted information
        system_message = """
You are a concise summarization assistant. Your task is to provide a 1-2 paragraph summary that covers the main points 
based on extracted information from a legal document. The summary should be clear, capturing key aspects in a concise manner.
"""
        
        user_message = """
** EXTRACTED INFORMATION TO SUMMARIZE **

Here is the extracted key information:
{formatted_info_all_chunks}

** QUESTIONS TO ADDRESS IN THE SUMMARY (if relevant information exists) **
{question_list}

** TASK **
Please summarize the information above into a 1-2 paragraph summary, ensuring that you address the given questions if relevant information exists within the extracted info.
"""

        # Create the prompt template with messages
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(user_message, input_variables=["formatted_info_all_chunks", "question_list"])
        ])

        # Generate the summary using the LLM
        chain = prompt | self.llm
        summary = chain.invoke({
            "formatted_info_all_chunks": formatted_info_all_chunks,
            "question_list": question_list,
        })

        if save_to_file:
            assert output_file is not None
            with open(output_file, 'w') as file:
                file.write("Summary:\n")
                file.write(summary.content + "\n\n")   
            print(f"Summary successfully written to {output_file}")

        if print_to_screen:
            print(f"********** Summary: **********\n{summary.content}\n********************\n")

        return summary.content


    def identify_quotes_for_summary(self, 
                             summary: str, 
                             extracted_info_all_chunks: List[QuestionAndAnswers],
                             save_to_file: bool = True,
                             print_to_screen: bool = False,
                             output_file: str = None,
                             output_format: str = 'yaml'):
        """
        Given a summary and extracted information, this method finds supporting short sentences from the extracted info
        for each sentence in the summary.
        
        Args:
            summary (str): The text summary to verify.
            extracted_info_all_chunks (List[QuestionAndAnswers]): Extracted information containing short sentences.
            save_to_file (bool): Whether to save the output to a file.
            print_to_screen (bool): Whether to print the output.
            output_file (str): The file path to save the output.
        
        Returns:
            dict: A dictionary where each key is a sentence from the summary and its value is a list of supporting ShortSentence objects.
        """
        # Split the summary into sentences
        summary_sentences = [sentence.strip() for sentence in summary.split('.') if sentence.strip()]

        # Format the extracted info for the prompt
        formatted_info = ""
        for chunk_index, extracted_info in enumerate(extracted_info_all_chunks):
            formatted_info += f"\n** CHUNK {chunk_index + 1}:** \n"
            for qa in extracted_info.response:
                formatted_info += f"Question: {qa.question}\n"
                for detail in qa.answers:
                    formatted_info += f"- {detail.short_sentence} (verbatim snippet: '{detail.short_verbatim_snippet}')\n"

        # System message to guide the LLM
        system_message = """
You are an assistant that helps verify the accuracy of summaries against extracted information from legal documents. 
Your task is to match sentences from the provided summary with supporting evidence from the extracted information.

For each sentence in the summary, identify which short sentences and verbatim snippets (from the extracted information) best support that summary sentence.

Please ensure that all supporting information comes directly from the provided extracted info and matches exactly.
"""

        # User message template
        user_message = """
**SUMMARY TO VERIFY:**

{summary}

**EXTRACTED INFORMATION TO USE:**

{formatted_info}

**TASK:**
For each sentence in the summary, identify and list the short sentences from the extracted information that best support that summary sentence.

**EXPECTED OUTPUT FORMAT:**
```json
{{
    "sentence 1": [
        {{
            "short_sentence": "a single-clause sentence that supports the summary sentence.",
            "short_verbatim_snippet": "A direct quote from the document supporting the summary sentence."
        }},
        ... more supporting sentences if available
    ],
    "sentence 2": [
        {{
            "short_sentence": "Another supporting sentence.",
            "short_verbatim_snippet": "Another supporting quote."
        }}
    ],
    ... for other sentences in the summary.
}}"""

        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(
                user_message, 
                input_variables=["summary", "formatted_info"]
            )
        ])

        # Create a chain with the LLM to generate structured output
        llm_with_structure = self.llm.with_structured_output(SentenceVerificationList)
        chain = prompt | llm_with_structure

        # Run the LLM to get the verification output
        verification_result = chain.invoke({
            "summary": summary,
            "formatted_info": formatted_info
        })

        # Convert the result to a dictionary for easier processing
        verification_dict = {}
        for verification in verification_result.verification:
            verification_dict[verification.summary_sentence] = [
                {
                    "short_sentence": sentence.short_sentence,
                    "short_verbatim_snippet": sentence.short_verbatim_snippet,
                    "direct_link": self.generate_link_to_snippet(
                            self.prefix_string, sentence.short_verbatim_snippet, self.clearinghouse_doc_id
                        )
                } for sentence in verification.supporting_sentences
            ]

        # Save to file if specified
        if save_to_file and output_file:
            assert output_format in ['yaml', 'json']
            if output_format == 'json':
                import json
                with open(output_file, 'w') as file:
                    json.dump(verification_dict, file, indent=4)
                    print(f"Verification results successfully written to {output_file}")
            else:
                import yaml
                # Save to file if specified
                if save_to_file and output_file:
                    with open(output_file, 'w') as file:
                        yaml.dump(verification_dict, file, default_flow_style=False, sort_keys=False, allow_unicode=True)
                        print(f"Verification results successfully written to {output_file}")

        # Print to screen if needed
        if print_to_screen:
            assert output_format in ['yaml', 'json']
            print("\n********** Verification Results **********")
            if output_format == 'json':
                # Print in JSON format
                print(json.dumps(verification_dict, indent=4))
            else:
                # Print in YAML format
                print(yaml.dump(verification_dict, default_flow_style=False, sort_keys=False, allow_unicode=True))
            print("*****************************************")


# # Instantiate the LLM
# llm = ChatOpenAI(model="gpt-4o-mini")

# # Read the input text from the specified file
# file_path = '/home/hice1/tnguyen868/scratch/openai-summarization/order_opinion_texts/doc_55784.txt'
# with open(file_path, 'r', encoding='utf-8') as file:
#     input_text = file.read()

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


# summarizer = Summarizer(
#     llm=llm,
#     input_text=input_text,
#     clearinghouse_doc_id=55784,
#     question_list=question_list,
#     document_type="order/opinion", 
#     short_sentence_length=11, 
#     num_short_sentences=20, 
#     quote_length=10,
# )

# extracted_info_all_chunks = summarizer.extract_sentences_for_questions()

# summarizer.print_extracted_info(extracted_info_all_chunks, generate_link=True, output_file="/home/hice1/tnguyen868/scratch/openai-summarization/order_opinion_texts/extracted_info_55784.txt")

# summary = summarizer.generate_concise_summary(extracted_info_all_chunks, 
#     question_list=question_list, 
#     output_file="/home/hice1/tnguyen868/scratch/openai-summarization/order_opinion_texts/summary_55784.txt")

# verification_results = summarizer.identify_quotes_for_summary(
#     summary=summary,
#     extracted_info_all_chunks=extracted_info_all_chunks,
#     output_file="/home/hice1/tnguyen868/scratch/openai-summarization/order_opinion_texts/summary_quotes_55784.txt"
# )

# pass
