import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

normal_repr = torch.Tensor.__repr__ 
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}" 

# code adapted from the NuExtract model's recommended usage: https://huggingface.co/numind/NuExtract-v1.5
MAX_INPUT_SIZE = 10_000
MAX_NEW_TOKENS = 2000

def clean_json_text(text):
    text = text.strip()
    text = text.replace("\#", "#").replace("\&", "&")
    return text

def predict_chunk(text, template, current, model, tokenizer):
    current = clean_json_text(current)

    input_llm =  f"<|input|>\n### Template:\n{template}\n### Current:\n{current}\n### Text:\n{text}\n\n<|output|>" + "{"
    input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=MAX_INPUT_SIZE).to("cuda")
    output = tokenizer.decode(model.generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS)[0], skip_special_tokens=True)

    return clean_json_text(output.split("<|output|>")[1])

def split_document(document, window_size, overlap):
    tokens = tokenizer.tokenize(document)
    print(f"\tLength of document: {len(tokens)} tokens")

    chunks = []
    if len(tokens) > window_size:
        for i in range(0, len(tokens), window_size-overlap):
            print(f"\t{i} to {i + len(tokens[i:i + window_size])}")
            chunk = tokenizer.convert_tokens_to_string(tokens[i:i + window_size])
            chunks.append(chunk)

            if i + len(tokens[i:i + window_size]) >= len(tokens):
                break
    else:
        chunks.append(document)
    print(f"\tSplit into {len(chunks)} chunks")

    return chunks

def handle_broken_output(pred, prev):
    try:
        if all([(v in ["", []]) for v in json.loads(pred).values()]):
            # if empty json, return previous
            pred = prev
    except:
        # if broken json, return previous
        pred = prev

    return pred

def sliding_window_prediction(text, template, model, tokenizer, window_size=4000, overlap=128):
    # split text into chunks of n tokens
    tokens = tokenizer.tokenize(text)
    chunks = split_document(text, window_size, overlap)

    # iterate over text chunks
    prev = template
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i}...")
        pred = predict_chunk(chunk, template, prev, model, tokenizer)

        # handle broken output
        pred = handle_broken_output(pred, prev)
            
        # iterate
        prev = pred

    return pred

nuExtract_dir = "/home/hice1/tnguyen868/scratch/openai-summarization/NuExtract_cache"
model_name = "numind/NuExtract-v1.5"
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, cache_dir=nuExtract_dir).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=nuExtract_dir)

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

complaint_summary_template = """{
    "Case Information": {
        "Filing Date": [],
        "Court Name": [],
        "Judge Name": [],
        "Counsel Type": [],
        "Class Action": [],
        "Defendants": [],
        "Plaintiffs": [],
        "Legal Claims": {
            "Statutory or Constitutional Basis": [],
            "State Claim": [],
            "Specific Allegations": []
        },
        "Remedies": {
            "Injunctive Relief": [],
            "Declaratory Relief": [],
            "Attorney Fees": [],
            "Money Damages": []
        }
    }
}"""


for i, complaint_idx in enumerate(document_IDs):
    doc = df["Document"].iloc[complaint_idx]
    # summary, metadata = summarize_doc(doc, doc_name=f"COMPLAINT #{complaint_idx} - \"{case_names[i]}\"", print_summary=True, print_token_usage=True, return_metadata=True)
    prediction = sliding_window_prediction(doc, complaint_summary_template, model, tokenizer, window_size=4000, overlap=128)
    print(f"COMPLAINT #{complaint_idx}")
    print(prediction)
    pass 
