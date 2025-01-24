from fastapi import FastAPI, UploadFile, Request, File
from mangum import Mangum
from .tools import GeminiTools, QdrantTools, PDFTools
app = FastAPI()

# Handles pdf input
@app.post("/chat/pdf", status_code=201)
async def file_upload(pdf_file: UploadFile = File(...)):
    try:
        # Read the PDF content
        # pdf_bytes = await pdf_file.read()
        # print(type(pdf_bytes))
        pdftool = PDFTools(pdf_file.file)
        print("hi 2")
        model = GeminiTools() # creating an instance of the model
        print('hi 3')
        conn = QdrantTools()  # connecting with qdrant cloud
        print('hi 4')
        conn.create_collection('new_collection2')
        print("hi 5")
        # has error
        html_doc = pdftool.extract_content() # Returns in html format for gemini to understand
        print('hi 6')
        chunks = model.generate_chunks(html_doc) # a list of chunk with their topic

        embeddings = GeminiTools.generate_embeddings(chunks)

        conn.store_information("new_collection2", embeddings, chunks)

        return {"message": "PDF file was uploaded"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An internal server error occurred.")

# handles prompts to the chat
@app.post("/chat/", status_code=200)
async def root(request: Request):
    try:
        model = GeminiTools()  # creating an instance of the model
        print('bye')
        conn = QdrantTools()  # connecting with qdrant cloud
        print('bye2')
        form_data = await request.form()
        print('bye3')
        prompt = form_data.get("message")
        print('bye4')
        embedding = GeminiTools.generate_embeddings(prompt)
        print('bye5')
        relevant_information = conn.retrieve("new_collection2", embedding)
        print('bye6')
        response = model.final_prompt(prompt, relevant_information)
        print('bye7')
        return {"Response": response}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,)



handler = Mangum(app)