import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
import io
from pdfminer.converter import HTMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

def dict_maker(string):
    return dict(chunk=string)

class GeminiTools:
    def __init__(self):
        genai.configure(api_key="AIzaSyD8Ua46J-4MARdmQRzBJiL15tK4FWlxYzk")
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")

    def generate_chunks(self, html_doc):
        chunks_string = self.model.generate_content(
            "Main goal: Chunk this text from a pdf which is in html format and add a token in the beginning of each chunk to" \
            " specify the common sub-topic in the document to which the chunk belongs.\n" \
            "Instructions: Make sure to include the information about dates in its associated chunk. \n" \
            "Dont include any other response other than the chunked texts which are separated by the " \
            "special token <chunk_separator>: \n\n" + html_doc
        )
        final_chunks = chunks_string.text.split('<chunk_separator>')
        return final_chunks

    def generate_embeddings(chunks):
        result = genai.embed_content(
            model= "models/text-embedding-004",
            content= chunks)
        return result['embedding']

    def final_prompt(self, prompt, retrieved_info):
        return self.model.generate_content(prompt + "Known Information : \n" + retrieved_info).text

class QdrantTools:
    def __init__(self):
        self.client = QdrantClient(
            url="https://8aa84678-6b00-47d8-af1e-78b993c0683b.eu-west-1-0.aws.cloud.qdrant.io:6333",
            api_key="hasdqBYTWjRoH7Li_s931shJyD-Jp16eP5YjEmN8F7HMPyLr9YZyOA",
        )

    def create_collection(self, name):
        self.client.create_collection(
            collection_name= name,
            vectors_config=VectorParams(size=768, distance=Distance.DOT),
        )

    def store_information(self, col_name, embeddings, final_chunks):

        final_chunks_dict = list(map(dict_maker, final_chunks))

        points = []
        for id, [vector, payload] in enumerate(zip(embeddings, final_chunks_dict)):
            points.append(PointStruct(id=id, vector=vector, payload=payload))

        operation_info = self.client.upsert(
            collection_name= col_name,
            wait=True,
            points=points
        )

        return operation_info

    # def delete_collection(self, name):
    #     pass

    # def get_collections(self):
    #     pass

    def retrieve(self, col_name, embedding):

        # call the search engine
        search_result = self.client.query_points(
            collection_name= col_name,
            query= embedding,
            with_payload=True,
            limit=3
        ).points

        retrieved_chunks = ""
        for item in search_result:
            retrieved_chunks += item.payload['chunk']

        return retrieved_chunks

class PDFTools:
    def __init__(self, file):
        self.file = file

    def extract_content(self):
        output_string = io.StringIO()
        parser = PDFParser(self.file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = HTMLConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

        return output_string.getvalue()















