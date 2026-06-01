import uvicorn

if __name__ == "__main__":
    # Ξεκινάει τον server τοπικά στη θύρα 8000 με ενεργοποιημένο το hot-reload
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)