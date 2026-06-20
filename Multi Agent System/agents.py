import os
from dotenv import load_dotenv  
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url

load_dotenv()

# model setup
llm = ChatMistralAI(
    model="mistral-medium-latest", 
    temperature=0, 
    max_tokens=2000, 
    api_key=os.getenv("MISTRAL_API_KEY")
)

# creation of 1st agent
def build_search_agent():
    # Modern approach: Bind the web_search tool directly to the Mistral LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research agent. Use your web search tool to find recent information. Always use the tool if you need information."),
        ("human", "{input}")
    ])
    return prompt | llm.bind_tools([web_search])

# creation of 2nd agent
def build_reader_agent():
    # Modern approach: Bind the scrape_url tool directly to the Mistral LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert reading agent. Use your scrape tool to gather full text content from a given URL link."),
        ("human", "{input}")
    ])
    return prompt | llm.bind_tools([scrape_url])

# creating writer chains
writer_prompt = ChatPromptTemplate.from_messages([
    ('system', "You are a expert research writer. Write clear ,strutured and insightful report."),
    ('human', """Write the detailed report on the given topic below.
Topics: {topics}
Research Gathered:
{research}
sturcture the report as:
-Introduction
-Key Findings(min 3 well explained points)
-Conclusion
-Sources(list the URLs of the sources used)
Be detailed functional and professional.""")
     
])

# how this chain works is that it takes the topics and research gathered as input and then passes it to the llm which generates the report based on the prompt given. The output of the llm is then parsed by the StrOutputParser() to get the final output in string format.
writer_chain = writer_prompt | llm | StrOutputParser()

# critic_chain =score the report weather it is correct or not and suggest what are the improovements.
critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()