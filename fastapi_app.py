from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from rag_pipeline import process_pdf_and_query
import uvicorn

app = FastAPI(title="DocuMind API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store uploaded PDFs temporarily
uploaded_pdfs = {}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file"""
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            pdf_path = tmp.name
        
        # Store the path
        uploaded_pdfs["current"] = {
            "path": pdf_path,
            "filename": file.filename
        }
        
        return {
            "status": "success",
            "message": f"✅ '{file.filename}' uploaded successfully!",
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/query")
async def query_pdf(question: str):
    """Query the uploaded PDF"""
    try:
        if "current" not in uploaded_pdfs:
            raise HTTPException(status_code=400, detail="👆 Please upload a PDF first!")
        
        pdf_path = uploaded_pdfs["current"]["path"]
        
        result = process_pdf_and_query(pdf_path, question)
        
        sources = []
        if result.get("sources"):
            for i, doc in enumerate(result["sources"]):
                page_num = doc.metadata.get('page', None)
                page_label = f"Page {page_num + 1}" if page_num is not None else f"Source {i+1}"
                content = doc.page_content if doc.page_content else "No content"
                sources.append({
                    "page": page_label,
                    "content": content[:300] + "..." if len(content) > 300 else content
                })
        
        return {
            "status": "success",
            "answer": result["answer"],
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.get("/")
async def root():
    """Serve the main HTML page"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🧠 DocuMind - AI-powered Document Intelligence</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                width: 100%;
                max-width: 700px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                padding: 40px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .header h1 {
                font-size: 2.5em;
                color: #667eea;
                margin-bottom: 10px;
            }
            
            .header p {
                color: #666;
                font-size: 1.1em;
            }
            
            .divider {
                height: 2px;
                background: #eee;
                margin: 30px 0;
            }
            
            .section {
                margin-bottom: 30px;
            }
            
            .section h2 {
                color: #333;
                font-size: 1.2em;
                margin-bottom: 15px;
            }
            
            .upload-area {
                border: 2px dashed #667eea;
                border-radius: 8px;
                padding: 30px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
                background: #f9f9ff;
            }
            
            .upload-area:hover {
                border-color: #764ba2;
                background: #f0f0ff;
            }
            
            .upload-area input {
                display: none;
            }
            
            .upload-area p {
                color: #666;
                margin-bottom: 10px;
            }
            
            .upload-btn {
                background: #667eea;
                color: white;
                padding: 10px 20px;
                border-radius: 6px;
                border: none;
                cursor: pointer;
                font-size: 1em;
                transition: background 0.3s;
            }
            
            .upload-btn:hover {
                background: #764ba2;
            }
            
            .file-info {
                margin-top: 15px;
                padding: 10px;
                background: #e8f5e9;
                border-radius: 6px;
                color: #2e7d32;
                display: none;
            }
            
            .chat-area {
                max-height: 400px;
                overflow-y: auto;
                margin-bottom: 20px;
                padding: 15px;
                background: #f9f9f9;
                border-radius: 8px;
                border: 1px solid #eee;
            }
            
            .message {
                margin-bottom: 15px;
                padding: 10px 15px;
                border-radius: 8px;
                word-wrap: break-word;
            }
            
            .user-message {
                background: #667eea;
                color: white;
                margin-left: 20px;
                text-align: right;
            }
            
            .assistant-message {
                background: #e8e8e8;
                color: #333;
                margin-right: 20px;
            }
            
            .input-group {
                display: flex;
                gap: 10px;
            }
            
            .input-group input {
                flex: 1;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 1em;
            }
            
            .send-btn {
                background: #667eea;
                color: white;
                padding: 12px 25px;
                border-radius: 6px;
                border: none;
                cursor: pointer;
                font-size: 1em;
                transition: background 0.3s;
            }
            
            .send-btn:hover {
                background: #764ba2;
            }
            
            .send-btn:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            
            .sources {
                margin-top: 15px;
                padding: 15px;
                background: #fff9e6;
                border-radius: 6px;
                border-left: 4px solid #ffc107;
            }
            
            .sources h4 {
                color: #856404;
                margin-bottom: 10px;
            }
            
            .source-item {
                margin-bottom: 10px;
                padding: 8px;
                background: white;
                border-radius: 4px;
                font-size: 0.9em;
                color: #666;
            }
            
            .source-item strong {
                color: #333;
            }
            
            .error {
                background: #ffebee;
                color: #c62828;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 15px;
                display: none;
            }
            
            .loading {
                display: none;
                text-align: center;
                color: #667eea;
                font-style: italic;
            }
            
            .info-box {
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                padding: 15px;
                border-radius: 4px;
                color: #1565c0;
                margin-bottom: 15px;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 DocuMind</h1>
                <p>AI-powered Document Intelligence — Upload a PDF and ask anything!</p>
            </div>
            
            <div class="divider"></div>
            
            <div class="section">
                <h2>📄 Step 1: Upload your PDF</h2>
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <p>📁 Click to upload or drag and drop</p>
                    <button class="upload-btn">Choose PDF</button>
                    <input type="file" id="fileInput" accept=".pdf" />
                </div>
                <div class="file-info" id="fileInfo"></div>
                <div class="error" id="uploadError"></div>
            </div>
            
            <div class="divider"></div>
            
            <div class="section">
                <h2>💬 Step 2: Ask anything from your PDF</h2>
                <div class="info-box" id="infoBox">👆 Please upload a PDF first!</div>
                
                <div class="chat-area" id="chatArea"></div>
                <div class="loading" id="loading">🔍 Searching document...</div>
                <div class="error" id="queryError"></div>
                
                <div class="input-group">
                    <input 
                        type="text" 
                        id="questionInput" 
                        placeholder="Ask a question about your document..." 
                        disabled
                    />
                    <button class="send-btn" id="sendBtn" disabled>Send</button>
                </div>
            </div>
        </div>
        
        <script>
            let pdfUploaded = false;
            
            // File upload handling
            document.getElementById('fileInput').addEventListener('change', uploadFile);
            document.getElementById('questionInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !document.getElementById('sendBtn').disabled) {
                    sendQuestion();
                }
            });
            document.getElementById('sendBtn').addEventListener('click', sendQuestion);
            
            async function uploadFile(event) {
                const file = event.target.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    document.getElementById('uploadError').style.display = 'none';
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        pdfUploaded = true;
                        document.getElementById('fileInfo').textContent = '✅ ' + data.message;
                        document.getElementById('fileInfo').style.display = 'block';
                        document.getElementById('infoBox').style.display = 'none';
                        document.getElementById('questionInput').disabled = false;
                        document.getElementById('sendBtn').disabled = false;
                    } else {
                        throw new Error(data.detail);
                    }
                } catch (error) {
                    document.getElementById('uploadError').textContent = '❌ ' + error.message;
                    document.getElementById('uploadError').style.display = 'block';
                }
            }
            
            async function sendQuestion() {
                const question = document.getElementById('questionInput').value.trim();
                if (!question) return;
                
                // Add user message to chat
                addMessage(question, 'user');
                document.getElementById('questionInput').value = '';
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('queryError').style.display = 'none';
                
                try {
                    const response = await fetch('/query?question=' + encodeURIComponent(question), {
                        method: 'POST'
                    });
                    
                    const data = await response.json();
                    document.getElementById('loading').style.display = 'none';
                    
                    if (response.ok) {
                        let answerHtml = data.answer;
                        if (data.sources && data.sources.length > 0) {
                            answerHtml += '<div class="sources"><h4>📚 Source Pages</h4>';
                            data.sources.forEach(source => {
                                answerHtml += '<div class="source-item"><strong>' + source.page + ':</strong> ' + source.content + '</div>';
                            });
                            answerHtml += '</div>';
                        }
                        addMessage(answerHtml, 'assistant');
                    } else {
                        throw new Error(data.detail);
                    }
                } catch (error) {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('queryError').textContent = '❌ ' + error.message;
                    document.getElementById('queryError').style.display = 'block';
                }
            }
            
            function addMessage(content, role) {
                const chatArea = document.getElementById('chatArea');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + (role === 'user' ? 'user-message' : 'assistant-message');
                messageDiv.innerHTML = content;
                chatArea.appendChild(messageDiv);
                chatArea.scrollTop = chatArea.scrollHeight;
            }
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
