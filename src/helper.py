from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from src.prompt import *
import time


# GoogleAi authentication
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY



def file_processing(file_path):

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen += page.page_content
        
    splitter_ques_gen = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size = 10000,
        chunk_overlap = 200
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size = 1000,
        chunk_overlap = 100
    )


    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path,output_file="data/answers.txt", max_qs=10, delay=5):
    """
    Complete pipeline:
    1. Process file → generate questions
    2. Build embeddings + vector store
    3. Setup QA chain
    4. Run answers for up to `max_qs` questions
    5. Save results to file with error handling
    Returns:
        qa_chain -> RetrievalQA object (answer generation chain)
        ques_list -> Cleaned list of generated questions
    """

    # Step 1: File processing (returns docs for Q & A)
    document_ques_gen, document_answer_gen = file_processing(file_path)

    # Step 2: Question generation LLM
    llm_ques_gen = ChatGoogleGenerativeAI(
        temperature=0.3,
        model="gemini-2.5-flash-lite"
    )

    PROMPT_QUESTIONS = PromptTemplate(
        template=prompt_template,
        input_variables=["text"]
    )
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    # Step 3: Generate questions
    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS
    )
    raw_questions = ques_gen_chain.run(document_ques_gen)

    # Clean questions
    ques_list = [q.strip() for q in raw_questions.split("\n") if q.strip()]

    # Step 4: Build vector store for answers
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    llm_answer_gen = ChatGoogleGenerativeAI(
        temperature=0.4,
        model="gemini-2.5-flash-lite"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )

        # Step 5: Run answers + save to file
    for idx, question in enumerate(ques_list[:max_qs], start=1):
        print(f"\n[{idx}] Question:", question)

        retry_count = 0
        max_retries = 3  # maximum retries per question
        while retry_count <= max_retries:
            try:
                response = qa_chain.invoke({"query": question})
                answer = response.get("result", "No answer returned.")

                print("Answer:", answer)

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write("Question: " + question + "\n")
                    f.write("Answer: " + answer + "\n\n")
                    f.write("--------------------------------------------------\n\n")
                break  # exit retry loop on success

            except Exception as e:
                retry_count += 1
                print(f"⚠️ Error processing question (attempt {retry_count}): {question}")
                print("Error details:", e)

                if retry_count > max_retries:
                    print("❌ Max retries reached. Skipping question.")
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write("Question: " + question + "\n")
                        f.write("Error: " + str(e) + "\n\n")
                        f.write("--------------------------------------------------\n\n")
                    break  # skip to next question
                else:
                    # optional: exponential backoff for retries
                    time.sleep(delay * 2 ** (retry_count - 1))

        # Respect quota (delay between calls)
        time.sleep(delay)

    print(f"\n✅ Finished processing {min(len(ques_list), max_qs)} questions.")
    return qa_chain, ques_list[:max_qs]


