from uvicorn import run

from app.main import app

if __name__ == "__main__":
    run(app, host="127.0.0.1", port=8000)
