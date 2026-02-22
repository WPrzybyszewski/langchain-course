from dotenv import load_dotenv


def main():
    print("Hello from langchain-course!")
    print(os.getnv("OPENAI_API_KEY"))


if __name__ == "__main__":
    main()
