import pickle

import backoff
import oss2
import requests
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, QuestionAnswerPrompt
from requests.auth import HTTPBasicAuth

from logger import logger


class OSS:
    endpoint = 'http://oss-cn-beijing.aliyuncs.com'
    # auth = oss2.Auth(os.getenv("OSSAK",""), os.getenv("OSSAS",""))
    auth = oss2.Auth("LTAI4G5qT2EAhSAxhQqqHjuH", "s5Toh00KLjLuug7OdmK3nBOkUF7B6Q")
    bucketName = "stoooges-test"
    bucket = oss2.Bucket(auth, endpoint, bucketName)

    @classmethod
    def upload(cls, path, fileName=None, fileObj=None, url=None, suffix=False, isPdf=False, headers=None, content=""):
        '''
        url/fileObj/content 三选一，url为通过url获取的文件流，fileObj为文件对象，content为文件字节流
        需要改名但不确定后缀时 suffix=True 想改后缀时suffix=xxx 否则不传suffix
        '''
        try:
            if url:
                content = requests.get(url)
                if not fileName:
                    fileName = url.split("/")[-1]
                else:
                    if suffix:
                        fileName = "{}.{}".format(fileName, url.split(".")[-1])
            elif content:
                fileName = "{}.{}".format(fileName, suffix)

            else:
                suffix = fileObj.filename.split(".")[-1]

                if fileName:
                    fileName = "{}.{}".format(fileName, suffix)
                else:
                    fileName = fileObj.filename

                content = fileObj.read()
            fullFileName = "{}/{}".format(
                path, fileName)
            # # 小桔：上传头像时，如果大小是5093，说明头像不能正常显示，如果是0，说明头像不存在
            # if path == "avatar" and len(content.content) in [5093, 0]:
            #     return False
            if isPdf == True:
                headers = {'Content-Type': 'application/pdf'}
            suffix = fullFileName.split(".")[-1]
            if suffix in ["jpg", "jpeg", "jpe", "jfif", "gif", "png"]:
                headers = {'Content-Type': 'image/jpg'}
            # elif suffix in ["gif", "png"]:
            #     headers = {'Content-Type': 'image/{}'.format(suffix)}
            elif suffix in ["doc"]:
                headers = {'Content-Type': 'application/msword'}
            elif suffix in ["docx"]:
                headers = {'Content-Type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'}
            elif suffix in ["mp3"]:
                headers = {'Content-Type': 'audio/mp3'}
            elif suffix in ["mp4"]:
                headers = {'Content-Type': 'video/mp4'}
            elif suffix in ["ppt"]:
                headers = {'Content-Type': 'application/vnd.ms-powerpoint'}
            elif suffix in ["xls"]:
                headers = {'Content-Type': 'application/vnd.ms-excel'}
            elif suffix in ["xlsx"]:
                headers = {'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
            elif suffix in ["exe"]:
                headers = {'Content-Type': 'application/x-msdownload'}
            cls.bucket.put_object(fullFileName, content, headers=headers)
            return fullFileName
        except Exception as e:
            logger.error("oss upload raise exception (path:{} fileName:{} url:{}):{}".format(path, fileName, url, e))
            return ""

    @classmethod
    def get(cls, objName):
        return cls.bucket.get_object(objName).read()

    @classmethod
    def exists(cls, objName):
        return cls.bucket.object_exists(objName)

    @classmethod
    def create_dir(cls, dir_full_path):
        """

        @param dir_full_path: str / 为文件夹的标识，一定要以/结尾
        @return:
        """
        if not cls.exists(dir_full_path) and dir_full_path:
            cls.bucket.put_object(dir_full_path, '')
            return True
        return False

    @classmethod
    def get_file_list_from_dir(cls, dir_full_path):
        """

        @param dir_full_path: str / 为文件夹的标识，一定要以/结尾
        @return:
        """
        if cls.exists(dir_full_path):
            obj_list = []
            num = 0
            for obj in oss2.ObjectIterator(cls.bucket, prefix=dir_full_path):
                if num == 0:
                    num += 1
                    continue
                obj_list.append(obj.key)
            return obj_list
        return False


# @backoff.on_exception(backoff.expo, Exception, max_tries=3)
def langchain_handle_single_document(in_file_path, prompt, output_parser, query):
    if in_file_path.endswith('.pdf'):
        loader = PyPDFLoader(in_file_path)
    elif in_file_path.endswith('.doc') or in_file_path.endswith('.docx'):
        loader = Docx2txtLoader(in_file_path)
    else:
        return None
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)

    qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=prompt)
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vectorstore.as_retriever())
    result = qa.run(query)
    result = output_parser.parse(result)
    del vectorstore
    return result


def llama_handle_single_pdf(file_path, prompt, query):
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    index = GPTVectorStoreIndex.from_documents(documents)

    QA_PROMPT = QuestionAnswerPrompt(prompt)
    query_engine = index.as_query_engine(text_qa_template=QA_PROMPT)
    response = query_engine.query(query)
    return response


def write_content_to_file(file_content, file_path, mode='wb'):
    """
    将文件内容写入指定文件中

    :param file_content: file content
    :param file_path: file path
    :param mode: 写入模式
    :return: None
    """
    with open(file_path, mode) as f:
        f.write(file_content)


def get_basic_auth_file(url, user, password, target_file_path):
    """
    取一个使用basic auth方式认证的文件，将其内容存入 target_file_path 中

    :param url:
    :param user:
    :param password:
    :param target_file_path:
    :return: None
    """
    response = requests.get(url, auth=HTTPBasicAuth(user, password))
    with open(target_file_path, 'wb') as f:
        f.write(response.content)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(in_data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(in_data, f)


if __name__ == '__main__':
    # 测试代码位置
    get_basic_auth_file('https://test.stoooges.cn/static/universityFiles/张元_UCAS_2023-24.pdf', 'Stoooges',
                        'Stoooges2020!', '/Users/chengyu/stoooges/gpt_test/data/张元_UCAS_2023-24.pdf')
