from bs4 import BeautifulSoup, NavigableString
import ast
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

def generate_xpath(element, path=""): # used to generate dict nodes
    """ Recursive function to generate the xpath of an element """
    if element.parent is None:
        return path
    else:
        siblings = [sib for sib in element.parent.children if sib.name == element.name]
        if len(siblings) > 1:
            count = siblings.index(element) + 1
            path = f"/{element.name}[{count}]{path}"
        else:
            path = f"/{element.name}{path}"
        return generate_xpath(element.parent, path)

def create_xpath_dict(html, only_body=True, max_length=200): # used to generate dict nodes
    ''' Create a list of xpaths and a list of dict of attributes of all elements in the html'''
    soup = BeautifulSoup(html, 'html.parser')
    if only_body:
        root = soup.body
    else:
        root = soup.html
    element_xpath_list = []
    element_attributes_list = []
    stack = [(root, '')]  # stack to keep track of elements and their paths
    while stack:
        element, path = stack.pop()
        if element.name is not None:
            current_path = generate_xpath(element)
            element_attrs = dict(element.attrs)
            direct_text_content = ''.join([str(content).strip() for content in element.contents if isinstance(content, NavigableString) and content.strip()])
            if direct_text_content:
                element_attrs['text'] = direct_text_content
                element_attrs['element'] = element.name
                for key in element_attrs:
                    if len(element_attrs[key]) > max_length:
                        element_attrs[key] = element_attrs[key][:max_length]
                element_xpath_list.append(current_path)
                element_attributes_list.append(element_attrs)
            elif element_attrs != {}:
                element_attrs['element'] = element.name
                for key in element_attrs:
                    if len(element_attrs[key]) > max_length:
                        element_attrs[key] = element_attrs[key][:max_length]
                element_xpath_list.append(current_path)
                element_attributes_list.append(element_attrs)
            for child in element.children:
                if child.name is not None:
                    stack.append((child, current_path))

    return element_xpath_list, element_attributes_list

def get_results(query, html, embedder, top_n=5, rank_fields=None): # used to generate and retrieve dict nodes
    ''' Return the top_n elements of the html that are the most relevant to the query as Node objects with xpath in their metadata'''
    xpath_list, attributes_list = create_xpath_dict(html)
    assert len(xpath_list) == len(attributes_list)
    l = len(xpath_list)
    for attr, xpath in zip(attributes_list, xpath_list):
        attr['xpath'] = xpath
    list_of_results = []
    for j in range(0, l, 1000):
        attr = attributes_list[j:j+1000]
        if rank_fields:
            rank_fields.append('xpath')
            attr = [{k: v for k, v in d.items() if k in rank_fields} for d in attr]
        attr = [d for d in attr if (not((len(list(d.keys()))==2) and (('element' in list(d.keys())) and 'xpath' in list(d.keys())))) or d=={}]
        nodes = []
        for d in attr:
            xpath = d.pop('xpath')
            nodes.append(TextNode(text=str(d), metadata={'xpath': xpath}))
        index = VectorStoreIndex(nodes, embed_model=embedder)
        retriever = BM25Retriever.from_defaults(index = index, similarity_top_k=top_n)
        results = retriever.retrieve(query)
        list_of_results += results
    list_of_results = sorted(list_of_results, key=lambda x: x.score, reverse=True)
    return list_of_results[:top_n]

def match_element(attributes, element_specs): # used to find chunk nodes corresponding to retrieve dict nodes
    ''' Return the index of the element in element_specs that matches the attributes of the element'''
    i=0
    for spec in element_specs:
        matches = True
        for key in spec:
            if key in attributes:
                if isinstance(attributes[key], list):
                    if not set(spec[key]).issubset(set(attributes[key])):
                        matches = False
                        break
                elif attributes[key] != spec[key]:
                    matches = False
                    break
            else:
                matches = False
                break
        if matches:
            return i
        i+=1
    return None

def return_nodes_with_xpath(nodes, xpaths, results_dict): # used to find chunk nodes corresponding to retrieve dict nodes
    ''' Return the chunk nodes that have an element that matches the attributes of the elements in results_dict'''
    returned_nodes = []
    for node in nodes:
        node.metadata['xpath'] = []
        node.metadata['element'] = []
        split_html = node.text
        soup = BeautifulSoup(split_html, 'html.parser')
        for element  in soup.descendants:
            try:
                attribute = element.attrs
                direct_text_content = ''.join([str(content).strip() for content in element.contents if isinstance(content, NavigableString) and content.strip()])
                if direct_text_content:
                    attribute['text'] = direct_text_content
                attribute['element'] = element.name
                indice = match_element(attribute, results_dict)
                if indice is not None:
                    node.metadata['xpath'].append(xpaths[indice])
                    node.metadata['element'].append(results_dict[indice])
                    returned_nodes.append(node)
            except:
                pass
    return returned_nodes

def get_nodes_sm(query, html, embedder, top_n=5, rank_fields=['element', 'placeholder', 'text', 'name']):
    text_list = [html]
    documents = [Document(text=t) for t in text_list]
    splitter = LangchainNodeParser(lc_splitter=RecursiveCharacterTextSplitter.from_language(
            language="html",
        ))
    #chunk nodes
    nodes = splitter.get_nodes_from_documents(documents)
    #dict nodes
    results = get_results(query, html, embedder=embedder, top_n=top_n, rank_fields = rank_fields)
    results_dict = [ast.literal_eval(r.text) for r in results]
    xpaths = [r.metadata['xpath'] for r in results]
    #find chunk nodes corresponding to retrieve dict nodes
    returned_nodes = return_nodes_with_xpath(nodes, xpaths, results_dict)
    return returned_nodes, results_dict

def get_nodes_sm_with_xpath(query, html, embedder, top_n=5, rank_fields=['element', 'placeholder', 'text', 'name']): # used to add xpaths to the returned nodes
    nodes, results_dict = get_nodes_sm(query, html, embedder, top_n, rank_fields)
    returned_nodes = []
    for node in nodes:
       returned_nodes.append(node.text + f"""\n
        Here is a list of some xpaths of element of previous text:
        {node.metadata['xpath']}
        \n\n
        """)
    return returned_nodes