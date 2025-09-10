from chroma_manager import ChromaManager

print("Инициализация Chroma...")
retriever = ChromaManager().retriever(3)

def search(question: str):
    print("=" * 100)
    print('Вопрос пользователя:', question)
    print("=" * 100)
    print('Найденые документы:')

    documents = retriever.invoke(input=question)
    print('Найдено:', len(documents))
    for document in documents:
        print("=" * 50)
        print('Индекс (вопрос документа):', document.page_content)
        print('Содержание (ответ документа (metadata)):', document.metadata['answer'], "\n")

def main():
    questions = [
        "List all bosses",
        "What is the final boss",
        "How to defeat the final boss",
        "What armor is best against the final boss"
    ]

    for question in questions:
        search(question)

if __name__ == "__main__":
    main()