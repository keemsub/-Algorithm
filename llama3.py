#https://velog.io/@judy_choi/LLaMA3-%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-RAG-%EA%B5%AC%EC%B6%95-Ollama-%EC%82%AC%EC%9A%A9%EB%B2%95-%EC%A0%95%EB%A6%AC
## Llama3-KO + RAG
import os
import warnings
warnings.filterwarnings("ignore")
#Text(PDF) Loader
from langchain_community.document_loaders import PyMuPDFLoader

# PyMuPDFLoader 을 이용해 PDF 파일 로드
loader = PyMuPDFLoader("labor_low.pdf")
pages = loader.load()
#TextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 문서를 문장으로 분리
## 청크 크기 500, 각 청크의 50자씩 겹치도록 청크를 나눈다
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
docs = text_splitter.split_documents(pages)
#Text Vector Embedding
from langchain.embeddings import HuggingFaceEmbeddings

# 문장을 임베딩으로 변환하고 벡터 저장소에 저장
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    #model_kwargs={'device':'cpu'},
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True},
)
#VectorStore(Chroma)
# 벡터 저장소 생성
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(docs, embeddings)

# 벡터 저장소 경로 설정
## 현재 경로에 'vectorstore' 경로 생성
vectorstore_path = 'vectorstore'
os.makedirs(vectorstore_path, exist_ok=True)

# 벡터 저장소 생성 및 저장
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=vectorstore_path)
# 벡터스토어 데이터를 디스크에 저장
vectorstore.persist()
print("Vectorstore created and persisted")
#Model
from langchain_community.chat_models import ChatOllama

# Ollama 를 이용해 로컬에서 LLM 실행
## llama3-ko-instruct 모델 다운로드는 Ollama 사용법 참조
model = ChatOllama(model="llama3-ko-instruct", temperature=0)
#Retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
#LangChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Prompt 템플릿 생성
template = '''친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘.":
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

# RAG Chain 연결
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Chain 실행
query = "연장근로수당에 대해 알려 줘"
answer = rag_chain.invoke(query)

print("Query:", query)
print("Answer:", answer)
