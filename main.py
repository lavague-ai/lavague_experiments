from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from lavague.action_engine import ActionEngine
from lavague.defaults import DefaultLLM, DefaultEmbedder

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser, CodeSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import VectorStoreIndex, Document
from langchain.docstore.document import Document as Doc
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import numpy as np
from tqdm import tqdm
from llama_index.core.retrievers import QueryFusionRetriever
import os
from llama_index.llms.azure_openai import AzureOpenAI

from lavague.action_engine import extract_first_python_code

import os
from llama_index.llms.azure_openai import AzureOpenAI

from lavague.prompts import DEFAULT_PROMPT

DEFAULT_PLAYWRIGHT_PROMPT = '''
Your goal is to write Playwright Python code to answer queries.

Your answer must be a Python markdown only.

Prefer User-Facing Attributes, Use text selectors, like text="Visible Text", to target elements by their visible text. 
You can also use Attributes like aria-label, aria-labelledby, role, etc., to target elements.
When user-facing attributes are not available or sufficient, Prefer class names and IDs that are meaningful and unlikely to change. 
Avoid using automatically generated, framework-specific, or obfuscated classes.
Utilize parent-child relationships to narrow down the element, especially when looking for elements within a specific section of the page

You can assume the following code has been executed:
```python
from playwright.async_api import async_playwright

playwright = await async_playwright().start()
browser = await playwright.chromium.connect_over_cdp("http://localhost:9222")
default_context = browser.contexts[0]

# Retrieve the first page in the context.
page = default_context.pages[0]
```

---

HTML:
<!DOCTYPE html>
<html>
<head>
    <title>Mock Search Page</title>
</head>
<body>
    <h1>Search Page Example</h1>
    <input id="searchBar" type="text" placeholder="Type here to search...">
    <button id="searchButton">Search</button>
    <script>
        document.getElementById('searchButton').onclick = function() {{
            var searchText = document.getElementById('searchBar').value;
            alert("Searching for: " + searchText);
        }};
    </script>
</body>
</html>

Query: Click on the search bar 'Type here to search...', type 'selenium', and press the 'Enter' key

Completion:
```python
# Let's proceed step by step.
# First we need to identify the component first, then we can click on it.

# Based on the HTML, the link can be uniquely identified using the ID "searchBar"
# Click on the search bar
search_bar = page.locator('#searchBar').first
await search_bar.click()

# Type 'selenium' into the search bar
await search_bar.type('selenium')

# Press the 'Enter' key
await page.keyboard.press('Enter')

```

---

HTML:
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Mock Page for Selenium</title>
</head>
<body>
    <h1>Welcome to the Mock Page</h1>
    <div id="links">
        <a href="#link1" id="link1">Link 1</a>
        <br>
        <a href="#link2" class="link">Link 2</a>
        <br>
    </div>
</body>
</html>

Query: Click on the title Link 1 and then click on the title Link 2

Completion:
```python
# Let's proceed step by step.
# First we need to identify the first component, then we can click on it. Then we can identify the second component and click on it.

# Based on the HTML, the first link the link can be uniquely identified using the ID "link1"
# Let's use this ID with playwright to identify the link
link1 = page.locator('#link1').first

# Then we click on the link
await link1.click()

# The other link can be uniquely identified using the class "link"
# Let's use this class to identify the link
link2 = page.locator('.link').first

# Click on the element found
await link2.click()
```

---

HTML:
<!DOCTYPE html>
<html>
<head>
    <title>Mock Page</title>
</head>
<body>
    <p id="para1">This is the first paragraph.</p>
    <p id="para2">This is the second paragraph.</p>
    <p id="para3">This is the third paragraph, which we will select and copy.</p>
    <p id="para4">This is the fourth paragraph.</p>
</body>
</html>

Query: Select the text inside the third paragraph

Completion:
```python
# Let's proceed step by step.

# Select the third paragraph element
third_paragraph = page.locator("(//p)[3]").first
# Get the text inside the third paragraph
text = third_paragraph.inner_text()
```

---

HTML:

Query: Scroll up a bit

Completion:
```python
# Let's proceed step by step.
# We don't need to use the HTML data as this is a stateless operation.
# 200 pixels should be sufficient. Let's execute the JavaScript to scroll up.

await page.evaluate("window.scrollBy(0, 200)")
```

---

---

HTML:
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Enhanced Mock Page for Selenium Testing</title>
</head>
<body>
    <h1>Enhanced Test Page for Selenium</h1>
    <div class="container">
        <button id="firstButton" onclick="alert('First button clicked!');">First Button</button>
        <!-- This is the button we're targeting with the class name "action-btn" -->
        <button class="action-btn" onclick="alert('Action button clicked!');">Action Button</button>
        <div class="nested-container">
            <button id="testButton" onclick="alert('Test Button clicked!');">Test Button</button>
        </div>
        <button class="hidden" onclick="alert('Hidden button clicked!');">Hidden Button</button>
    </div>
</body>
</html>


Query: Click on the Button 'Action Button'

Completion:
```python
# Let's proceed step by step.
# First we need to identify the button first, then we can click on it.

# Based on the HTML provided, we need to devise the best strategy to select the button.
# The action button can be identified using the class name "action-btn"
action_button = page.locator('.action-btn').first

# Then we can click on it
await action_button.click()
```

---

HTML:
{context_str}
Query: {query_str}
Completion:
'''

# api_key=os.getenv("AZURE_OPENAI_KEY")
# api_version="2023-05-15"
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# model = "gpt-4"
# deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4-turbo")

# class LLM(AzureOpenAI):
#     def __init__(self):
#         super().__init__(
#             model=model,
#             deployment_name=deployment_name,
#             api_key=api_key,
#             azure_endpoint=azure_endpoint,
#             api_version=api_version,
#             temperature=0.0
#         )
# llm = LLM()

api_key=os.getenv("AZURE_OPENAI_KEY")
api_version="2024-02-15-preview"
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model = "gpt-35-turbo"
deployment_name = "gpt-35-turbo"

class LLM(AzureOpenAI):
    def __init__(self):
        super().__init__(
            model=deployment_name,
            deployment_name=deployment_name,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            temperature=0.0
        )
llm = LLM()

embedder = DefaultEmbedder()


def get_retriever_code(embed, html):
    
    K = 3
    
    text_list = [html]
    documents = [Document(text=t) for t in text_list]
    
    splitter = CodeSplitter(
        language="html",
        chunk_lines=50,  # lines per chunk
        chunk_lines_overlap=15,  # lines overlap between chunks
        max_chars=2000,  # max chars per chunk
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    nodes = [node for node in nodes if node.text]

    index = VectorStoreIndex(nodes, embed_model=embed)
    retriever_code = BM25Retriever.from_defaults(index = index, similarity_top_k=K)
    return retriever_code

def get_retriever_recursive(embed, html):
    
    K = 2
    
    text_list = [html]
    documents = [Document(text=t) for t in text_list]
    
    splitter = LangchainNodeParser(lc_splitter=RecursiveCharacterTextSplitter.from_language(
        language="html",
    ))
    
    nodes = splitter.get_nodes_from_documents(documents)
    nodes = [node for node in nodes if node.text]

    index = VectorStoreIndex(nodes, embed_model=embed)
    retriever_recursive = BM25Retriever.from_defaults(index = index, similarity_top_k=K)
    return retriever_recursive

action_engine = ActionEngine(llm, embedder, streaming=False, prompt_template=DEFAULT_PLAYWRIGHT_PROMPT)
# get_retriever = get_retriever_recursive
get_retriever = get_retriever_code

app = FastAPI()

class InputData(BaseModel):
    query: str
    HTML: str

from pydantic import BaseModel
from typing import List, Dict, Any

class OutputData(BaseModel):
    code: str
    retrieved_nodes: List[str]
    metadata: Dict[str, Any]


@app.post("/process", response_model=OutputData)
async def process(input_data: InputData):
    # Example processing - Replace this with your actual logic
    query = input_data.query
    html = input_data.HTML
    
    code, retrieved_nodes = action_engine.get_action(query, html)

    return OutputData(code=code, retrieved_nodes=retrieved_nodes)

from llama_index.core import get_response_synthesizer
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
import time



def get_nodes(query, html):    
    print("Using recursive retriever")
    retriever = get_retriever(embedder, html)
    source_nodes = retriever.retrieve(query)
    source_nodes = [node.text for node in source_nodes]
    return source_nodes

@app.post("/process_fat_node", response_model=OutputData)
async def process_fat_node(input_data: InputData):
    # Example processing - Replace this with your actual logic
    query_str = input_data.query
    html = input_data.HTML
    
    start_time = time.time()

    # Your code here
    source_nodes = get_nodes(query_str, html)

    end_time = time.time()
    indexing_time = end_time - start_time
    print("Indexing time: ", indexing_time)   
    
    context_str = source_nodes[0] = "\n".join(source_nodes)
    prompt = DEFAULT_PLAYWRIGHT_PROMPT.format(context_str=context_str, query_str=query_str)
    # prompt = DEFAULT_PROMPT.format(context_str=context_str, query_str=query_str)
    
    start_time = time.time()

    # Your code here
    response = llm.complete(prompt).text

    end_time = time.time()
    completion_time = end_time - start_time
    print("Completion time: ", completion_time)
    
    code = extract_first_python_code(response)
    
    import inspect
    retriever_code = inspect.getsource(get_retriever)
    
    metadata = {
        "model_id": model,
        "retrieve_code": retriever_code,
        "indexing_time": indexing_time,
        "completion_time": completion_time,
        "prompt": prompt
    }
    
    return OutputData(code=code, retrieved_nodes=source_nodes, metadata=metadata)

@app.post("/process_direct", response_model=str)
async def process_direct(input_data: InputData):
    # Example processing - Replace this with your actual logic
    query = input_data.query
    html = input_data.HTML
    
    prompt = DEFAULT_PLAYWRIGHT_PROMPT.format(context_str=html, query_str=query)
    
    response = llm.complete(prompt)
    code = response.text
    return code

@app.post("/get_index", response_model=OutputData)
async def get_index(input_data: InputData):
    # Example processing - Replace this with your actual logic
    query = input_data.query
    html = input_data.HTML
    
    source_nodes = get_nodes(query, html)
    
    code = ""
    return OutputData(code=code, retrieved_nodes=source_nodes)
