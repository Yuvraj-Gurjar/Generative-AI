from langchain.tools import tool
# this is bcz we request varous websites for data
import requests
from bs4 import BeautifulSoup 
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from rich import print
load_dotenv()

# load tavilyclient api key
tavily=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# kya chahiye web se
@tool
def web_search(query:str)->str:
    """Search the web for recent and reliable imformation on a topic.Returns Titles,URLs and snippets."""
    # to search some result call tavily fun
    results=tavily.search(query=query,max_results=5) #to save tokens we use max_results
    # to see the results u can run these two lines
#     return results
# print(web_search.invoke("what are the recent news about the war (US vs IRAN)"))
    out =[]
    for r in results["results"]:
        out.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
        )
# join to get the string back
    return "\n------\n".join(out)    
# print(web_search.invoke("what is the recent news of US VS IRAN war"))    

# till here we have created a tool to search the web and get the recent news about any topic. we can use this tool in our agent to get the recent news about any topic. we can also use this tool to get the recent news about the war (US vs IRAN) by calling the web_search.invoke() function with the query "what is the recent news of US VS IRAN war".

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"
# print(scrape_url.invoke("https://www.bbc.com/news/world-middle-east-66707305"))        

# till here we have created a tool to scrape a given URL and return the clean text content from that URL. we can use this tool in our agent to get the clean text content from any URL by calling the scrape_url.invoke() function with the URL as the argument.

# now we have created two tools, one for web search and another for scraping a URL. we can use these tools in our agent to get the recent news about any topic and to get the clean text content from any URL. we can also use these tools together to get the recent news about the war (US vs IRAN) and then scrape the URLs of the news articles to get the clean text content from those articles.



# tool.py--->agents.py--->pipeline.py--->