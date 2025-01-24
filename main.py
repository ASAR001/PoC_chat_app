from fastapi import FastAPI, UploadFile, Request, File
from mangum import Mangum
from .tools import GeminiTools, QdrantTools, PDFTools
app = FastAPI()

# Handles pdf input
@app.post("/chat/pdf", status_code=201)
async def file_upload(pdf_file: UploadFile = File(...)):
    try:
        pdftool = PDFTools(pdf_file.file)
        model = GeminiTools() # creating an instance of the model
        conn = QdrantTools()  # connecting with qdrant cloud
        conn.create_collection('new_collection2')
        html_doc = pdftool.extract_content() # Returns in html format for gemini to understand
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
        conn = QdrantTools()  # connecting with qdrant cloud
        form_data = await request.form()
        prompt = form_data.get("message")
        embedding = GeminiTools.generate_embeddings(prompt)
        relevant_information = conn.retrieve("new_collection2", embedding)
        response = model.final_prompt(prompt, relevant_information)
        return {"Response": response}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,)



handler = Mangum(app)