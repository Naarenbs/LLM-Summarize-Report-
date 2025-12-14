from langchain_core.prompts import ChatPromptTemplate

def summarize_action(llm, context):
    """
    Summarizes the provided context.
    """
    prompt = ChatPromptTemplate.from_template(
        "You are an expert researcher. Provide a comprehensive summary of the document content below.\n"
        "Capture the main arguments, key details, and improved context.\n\n"
        "Text:\n{context}"
    )
    chain = prompt | llm
    return chain.invoke({"context": context})

def report_action(llm, context):
    """
    Generates a structured report based on the context.
    """
    prompt = ChatPromptTemplate.from_template(
        "Write a formal report based on the following text. "
        "The report must include a Title, Key Points (bulleted), and a Conclusion.\n\n{context}"
    )
    chain = prompt | llm
    return chain.invoke({"context": context})

def categorize_action(llm, context):
    """
    Categorizes the findings in the context.
    """
    prompt = ChatPromptTemplate.from_template(
        "Analyze the following text and categorize the main topics/themes.\n\n{context}"
    )
    chain = prompt | llm
    return chain.invoke({"context": context})