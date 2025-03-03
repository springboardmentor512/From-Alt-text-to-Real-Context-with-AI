from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain.tools.render import render_text_description

@tool # tool information
def get_length_of_string(string: str) -> int:
    """Returns the length of the string by characters"""
    print(f"Getting length of string: {string}")
    text = string.strip("'\n'").strip('"')
    return len(text)

if __name__ == "__main__":
    # llm intialization
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
    )

    # list of tools - n no of tools
    tools = [ get_length_of_string ]

    # React - Reason + Action
    template = """
        Answer the following questions as best you can.
        You have access to the following tools:
        {tools}
        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action (just the value, no function call syntax)
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: """
    
    # here we can't directly pass tools as it is array as llm receives only text as input so we can use render_text_description
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools])
    )

    chain = {"input": lambda x: x["input"]} | prompt | llm

    res = chain.invoke(
        {"input": "What is the length in characters of the text 'Taj Mahal'?" }
    )

    print(res)