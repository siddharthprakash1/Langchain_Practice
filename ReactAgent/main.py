from dotenv import load_dotenv
from langchain.agents import tool

@tool#by doing this you make it a structured tool in the langchian
def get_text_length(text:str)->int:
    """Returns the length og a text by charecters"""
    print(f"get_text_length enter with {text=}")
    text=text.strip("'\n").strip(
        '"'
    )#stripping away non alphabetic charecters just in case 
    return len(text)


if __name__=="__main__":
    print("Hello React Agent")
    tools=[get_text_length]
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought:
    """
    
    