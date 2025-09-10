from config import OLLAMA_URL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
from langchain_ollama import OllamaLLM
from chroma_manager import ChromaManager
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

langfuse = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)
langfuse_handler = CallbackHandler()

print("Инициализация LLM...")
llm = OllamaLLM(model='llama3.1:8b', base_url=f"{OLLAMA_URL}", temperature=0.15, num_predict=1024, reasoning=False)

print("Инициализация Chroma...")
retriever = ChromaManager().retriever()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "{context}"
    )

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt, document_prompt=PromptTemplate.from_template("{answer}"))
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []
def ask(question: str):
    print("=" * 100)
    print('Вопрос пользователя:', question)

    result = {'question': question, 'answer': ''}
    
    print("=" * 100)
    print('Ответ модели:')
    for chunk in rag_chain.stream({"input": question, "chat_history": chat_history}, config={"callbacks":[langfuse_handler]}):
        if 'answer' in chunk:
            print(chunk['answer'], end='', flush=True)
            result['answer'] += chunk['answer']
    print()

    chat_history.append(('human', result['question']))
    chat_history.append(('ai', result['answer']))
    

def main():
    questions = [
        # 'Какие есть боссы в Террарии?',
        # 'Какой финальный босс?',
        # 'И как его победить?',
        # 'Какую броню на него использовать?',

        'What bosses are there in Terraria?',
        'What is the final boss?',
        'And how to defeat it?',
        'What armor should be used against it?'
    ]

    for question in questions:
        ask(question)

if __name__ == "__main__":
    main()