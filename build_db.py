import os
# 配置 HuggingFace 镜像，防止下载模型卡住
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. 配置你的智谱 API Key
os.environ["ZAI_API_KEY"] = "你的智谱API_Key填在这里"

# 2. 定义你要加载的简历字典（键是标签，值是文件名）
resume_files = {
    "测试岗位": "my_resume.pdf", #简历文件路径
    "开发岗位": "my_resume_dev.pdf"
}

all_chunks = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)

print("开始处理简历文件...")
for tag, file_name in resume_files.items():
    if os.path.exists(file_name):
        print(f"正在读取：{file_name} (标签: {tag})")
        loader = PyPDFLoader(file_name)
        pages = loader.load_and_split()
        chunks = text_splitter.split_documents(pages)
        
        # ⭐ 核心步骤：给每个文本块打上标签
        for chunk in chunks:
            chunk.metadata["resume_type"] = tag
            
        all_chunks.extend(chunks)
    else:
        print(f"⚠️ 找不到文件：{file_name}，已跳过。")

print(f"总共切分出了 {len(all_chunks)} 个片段。")

# 3. 向量化并存入数据库
print("正在加载本地向量模型...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 先清理旧的数据库，防止重复添加
if os.path.exists("./chroma_db"):
    import shutil
    shutil.rmtree("./chroma_db") 
    print("已清理旧的数据库文件夹。")

print("正在构建新的多版本数据库...")
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("🎉 多版本数据库构建成功！")