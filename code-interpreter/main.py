from dotenv import load_dotenv
from langchain import hub
from langchain_community.llms import Ollama
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool


def main():
    print("Start...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    The qrcode package is already installed. Do not attempt to install it again.
    Always follow this exact format for your responses:

    Thought: <your reasoning>
    Action: Python_REPL
    Action Input: <your Python code>

    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    Execute the code in small, verifiable steps.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]

    # Initialize model using Ollama
    llm = Ollama(model="pxlksr/opencodeinterpreter-ds")

    agent = create_react_agent(
        prompt=prompt,
        llm=llm,
        tools=tools,
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    agent_executor.invoke(
        input={
            "input": "Generate and save in the current working directory 15 QR codes that point to https://siddharthprakashdev-siddharth-prakashs-projects.vercel.app/. The qrcode package is already installed. Do not attempt to install it.Donot just generate the code execute it and save the qr code."
        }
    )

if __name__ == "__main__":
    main()

