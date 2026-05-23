import uvicorn
from src.api import app

if __name__ == "__main__":
    # Ξεκινάει τον server τοπικά στη θύρα 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)