
# Date: 09/03/23
# Install the required packages
# openai, langchain, pypdf

#################### Summary using Langchain + OpenAI #################################
# Data preprocessing
import openai, langchain
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("kargil_war.pdf")
docs = loader.load_and_split()
filepath = r"C:\Users\Public\Projects\openai\projects\langchain\SummaryUseCase\generated_summary.txt"

# Define the template
from langchain.prompts import PromptTemplate
prompt_template = """
Generate summary for the following context. Format the response in points.
context:
{text}
"""
prompt = PromptTemplate.from_template(prompt_template)

# Define LLM
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

llm = ChatOpenAI(temperature = 0, model_name = "gpt-3.5-turbo-16k")
llm_chain = LLMChain(llm = llm, prompt = prompt)

# Stuff Documentschain
# Run the LLM
# stuff_chain = StuffDocumentsChain(llm_chain = llm_chain, document_variable_name = "text") #processing the docs
# response = stuff_chain.run(docs) #Run the LLM on docs

# with open(filepath,'w') as fobj:
#    fobj.write(response)


#################### OpenAI #################################
import pypdf
from pypdf import PdfReader
reader = PdfReader("kargil_war.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"
    
prompt = [{'role':'system', 'content':'generate concise summary'},
          {'role':'user', 'content': text}]

# response = openai.ChatCompletion.create(model = "gpt-3.5-turbo-16k", messages = prompt, temperature = 0, max_tokens = 2000)

# with open(filepath,'w') as fobj:
#    fobj.write(response["choices"][0]["message"]["content"])

#################### Longer document using OpenAI #################################

chunks = []
max_length = 5000
while len(text) > max_length:
    chunk_length = text[:max_length].rfind(' ')
    chunks.append(text[:chunk_length])
    text = text[chunk_length + 1:]
chunks.append(text)

combined_summary = ""
for chunk in chunks:
    prompt = [{'role':'system', 'content':'generate concise summary'},
            {'role':'user', 'content': chunk}]

    response = openai.ChatCompletion.create(model = "gpt-3.5-turbo-16k", messages = prompt, temperature = 0, max_tokens = 2000)
    combined_summary += response["choices"][0]["message"]["content"]

prompt = [{'role':'system', 'content':'generate concise summary. Format the summary points nicely.'},
            {'role':'user', 'content': combined_summary}]
response = openai.ChatCompletion.create(model = "gpt-3.5-turbo-16k", messages = prompt, temperature = 0, max_tokens = 2000)

with open(filepath,'w') as fobj:
    fobj.write(response["choices"][0]["message"]["content"])


