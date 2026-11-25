"""Single-command runner for AID201A Classical AI Document QA System.

Runs FastAPI with Uvicorn on http://127.0.0.1:8000
"""

import sys
import uvicorn

if __name__ == "__main__":
    print("==================================================================")
    print(" AID201A Classical AI Document Question Answering System")
    print(" Symbolic Knowledge Representation · Rule Inference · Graph Search")
    print(" Starting server on http://127.0.0.1:8000")
    print("==================================================================")
    uvicorn.run("app.backend.main:app", host="127.0.0.1", port=8000, reload=False)
