"""python -m voyager.portal  ->  serve the portal on http://127.0.0.1:8050"""
import uvicorn


def main() -> None:
    uvicorn.run("voyager.portal.app:app", host="127.0.0.1", port=8050, log_level="warning")


if __name__ == "__main__":
    main()
