from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Indexing (Step-1(a))
video_id = "tzrwxLNHtRY"
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id,languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    # print(transcript)
except TranscriptsDisabled:
    print("No caption available for this video.")

#  Indexing(Step-1(b)) -->Text splitting
splitter = RecursiveCharacterTextSplitter(chunk_size =1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
# print((chunks))

# Step 1(c and d)
embeddings = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2')
vector_store = FAISS.from_documents(chunks,embeddings)
# print(vector_store.index_to_docstore_id)

# Step 2(Retrival)
retriver = vector_store.as_retriever(search_type = 'similarity',search_kwargs ={"k":4})
# print(retriver.invoke('What is MCP?'))

# Step 3(Augmentation)
Prompt = PromptTemplate(
    template= """You are a helpful assistant.
    Answer only from the provided transcript context.
    If the context is unsufficient ,just say you don't know.
    {context}
    Question:{question}
    """,
    input_variables=['context' , 'question']
)
question ="Is the topic MCP disscussed in this video ? if yes then what is that?expain in detail."
Answer = retriver.invoke(question)
# print(Answer)
context_text = "\n\n".join(doc.page_content for doc in Answer)
# print(context_text)
final_prompt = Prompt.invoke({"context":context_text,"question":question})
# print(final_prompt)

# Step 4(Generation)
# llm = ChatGoogleGenerativeAI(model ='gemini-2.5-pro',temperature=0.2)
# Final_response = llm.invoke(final_prompt)
llm = HuggingFaceEndpoint(
    task = "text-generation",
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.2,
)

Final_response = llm.invoke(final_prompt)

# print(Final_response) but is providing meta data
print(Final_response)