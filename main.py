import itertools
import os
import random
import sqlite3 as sl
from io import StringIO
from tempfile import NamedTemporaryFile

import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import QAGenerationChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from pdf2image import convert_from_path  # type: ignore
from pytesseract import image_to_string  # type: ignore
from translate import Translator

st.set_page_config(page_title="Chat With PDF", page_icon=":robot_face:")


@st.cache_data
def load_docs(files, pdf_image):
    # st.info("`Processing ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            with NamedTemporaryFile(dir=".", suffix=".pdf") as f:
                print(f.write(file_path.getbuffer()))

            if pdf_image == False:
                pdf_reader = PyPDF2.PdfReader(file_path)
                print(file_path)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                all_text += text

            else:
                print("Processing OCR ....>>>")

                def convert_pdf_to_img(pdf_file):
                    return convert_from_path(pdf_file)

                def convert_image_to_text(file):
                    text = image_to_string(file)
                    return text

                def get_text_from_any_pdf(pdf_file):
                    images = convert_pdf_to_img(pdf_file)
                    final_text = ""
                    for pg, img in enumerate(images):

                        final_text += convert_image_to_text(img)

                    return final_text

                with NamedTemporaryFile(dir=".", suffix=".pdf") as f:
                    f.write(file_path.getbuffer())
                    print(get_text_from_any_pdf(f.name))
                    all_text = get_text_from_any_pdf(f.name)

        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning("Please provide txt or pdf.", icon="⚠️")
    return all_text


@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    try:
        vectorstore = FAISS.from_texts(splits, _embeddings)
    except (IndexError, ValueError) as e:
        st.error(f"Error creating vectorstore: {e}")
        return
    retriever = vectorstore.as_retriever(k=5)

    return retriever


@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    # Split texts
    # IN: text, chunk size, overlap, split_method
    # OUT: list of str splits

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits


@st.cache_data
def generate_eval(text: str, N: int, chunk: int) -> list:
    """
    Generate a set of evaluation questions by extracting chunks of text from a given input and using a language model
    to generate questions and answers. It then translates the questions and answers from English to French using a
    translation API.

    Args:
        text (str): The input text from which evaluation questions will be generated.
        N (int): The number of questions to generate.
        chunk (int): The size of the chunks of text to draw questions from in the input.

    Returns:
        list: A list of dictionaries containing the generated evaluation questions and answers, translated from English to French.
    """

    n = len(text)
    starting_indices = [random.randint(0, n - chunk) for _ in range(N)]
    sub_sequences = [text[i : i + chunk] for i in starting_indices]
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    chain = QAGenerationChain.from_llm(llm)
    eval_set = []

    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
        except:
            print("Error generating question %s." % str(i + 1))

    eval_set_full = list(itertools.chain.from_iterable(eval_set))

    translator = Translator(to_lang="fr")

    for item in eval_set_full:
        messages1 = [
            SystemMessage(
                content="You are a helpful assistant that translates English to French."
            ),
            HumanMessage(
                content="Translate this sentence from English to French:" + item["question"]
            ),
        ]
        messages2 = [
            SystemMessage(
                content="You are a helpful assistant that translates English to French."
            ),
            HumanMessage(
                content="Translate this sentence from English to French:" + item["answer"]
            ),
        ]
        result1 = llm.generate([messages1])
        result2 = llm.generate([messages2])
        item["question"] = result1.generations[0][0].text
        item["answer"] = result2.generations[0][0].text

    return eval_set_full


# ...


def main():

    foot = """
    <div style="
        position: fixed;
        bottom: 0;
        left: 30%;
        right: 0;
        width: 50%;
        padding: 0px 0px;
        text-align: center;
    ">

    </div>
    """

    st.markdown(foot, unsafe_allow_html=True)

    # Add custom CSS
    st.markdown(
        """
        <style>

        #MainMenu {visibility: hidden;
        # }
            footer {visibility: hidden;
            }
            .css-card {
                border-radius: 0px;
                padding: 30px 10px 10px 10px;
                color:black;
                background-color: #f8f9fa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 10px;
                font-family: "IBM Plex Sans", sans-serif;
            }

            .card-tag {
                border-radius: 0px;
                padding: 1px 5px 1px 5px;
                margin-bottom: 10px;
                position: absolute;
                left: 0px;
                top: 0px;
                font-size: 0.6rem;
                font-family: "IBM Plex Sans", sans-serif;
                color: white;
                background-color: green;
                }

            .css-zt5igj {left:0;
            }

            span.css-10trblm {margin-left:0;
            }

            div.css-1kyxreq {margin-top: -40px;
            }

            div.css-1y4p8pa { margin-top: -20px;

            }





        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.image("img/left.png")
    show_auto_questions = st.sidebar.checkbox("Questions générées")
    show_history = st.sidebar.checkbox("Historique des questions")

    st.image("img/jangat.jpeg")
    # st.write(
    # f"""
    # <div style="display: flex; align-items: center; margin-left: 0;">
    #     <h2 style="display: inline-block;">Chat With PDF</h2>
    # </div>
    # """,
    # unsafe_allow_html=True,
    #     )

    retriever_type = "SIMILARITY SEARCH"

    # openai_api_key = os.getenv("OPENAI_API_KEY")
    import os
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    #load_dotenv()
    uploaded_files = st.file_uploader(
        "Upload a PDF or TXT Document", type=["pdf", "txt"], accept_multiple_files=True
    )
    pdf_image = st.checkbox(
        "Cochez cette case si le PDF est basé sur une image/scanné.", value=False
    )

    if uploaded_files:
        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if (
            "last_uploaded_files" not in st.session_state
            or st.session_state.last_uploaded_files != uploaded_files
        ):
            st.session_state.last_uploaded_files = uploaded_files
            if "eval_set" in st.session_state:
                del st.session_state["eval_set"]

        # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(uploaded_files, pdf_image)
        st.write("Document chargé et traité.")

        # Split the document into chunks
        splits = split_texts(
            loaded_text, chunk_size=1000, overlap=0, split_method="RecursiveCharacterTextSplitter"
        )

        # Embed using OpenAI embeddings
        embeddings = OpenAIEmbeddings()
        retriever = create_retriever(embeddings, splits, retriever_type)

        # Initialize the RetrievalQA chain with streaming output
        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        chat_openai = ChatOpenAI(
            streaming=True, callback_manager=callback_manager, verbose=True, temperature=0
        )
        qa = RetrievalQA.from_chain_type(
            llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True
        )

        # Check if there are no generated question-answer pairs in the session state
        if "eval_set" not in st.session_state:
            # Use the generate_eval function to generate question-answer pairs
            num_eval_questions = 10  # Number of question-answer pairs to generate
            st.session_state.eval_set = generate_eval(loaded_text, num_eval_questions, 3000)

        st.write("Prêt à répondre à vos questions.")

        # Display the question-answer pairs in the sidebar with smaller text

        if show_auto_questions:
            st.sidebar.subheader("Questions générées")

            for i, qa_pair in enumerate(st.session_state.eval_set):
                with st.sidebar.expander(qa_pair["question"]):
                    st.write(qa_pair["answer"])

        # db
        con = sl.connect("history.db")
        cursor = con.cursor()

        # Question and answering
        with st.form("my_form", clear_on_submit=True):
            user_question = st.text_area("Saisissez votre question:")
            submitted = st.form_submit_button("Analyser")

        button_clicked = st.button("Clear")

        if show_history:
            st.sidebar.subheader("History")
            cursor.execute("SELECT * FROM QA ORDER BY ID DESC")
            rows = cursor.fetchall()

            for row in rows:
                question = row[1]
                answer = row[2]

                with st.sidebar.expander(question):
                    st.write(answer)

        if button_clicked:
            submitted = False
        if user_question:
            answer = qa.run(user_question)
            if submitted:
                st.subheader("Question")
                st.write(user_question)
                st.subheader("Réponses:")
                st.write(answer)

                try:
                    cursor.execute(
                        "INSERT INTO QA (question, ans) VALUES (?, ?)", (user_question, answer)
                    )
                except:
                    with con:
                        con.execute(
                            """
                            CREATE TABLE QA (
                            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                            question TEXT,
                            ans TEXT
                             );
                        """
                        )
                con.commit()
                con.close()


if __name__ == "__main__":
    main()
