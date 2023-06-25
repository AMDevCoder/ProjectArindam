import streamlit as st
import numpy as np
import joblib
from nltk import pos_tag, word_tokenize
from googlesearch import search
import nltk
import base64
import openai
import time

# Set up OpenAI API
openai.api_key = "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"

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load the trained model
model = joblib.load('trained_model.joblib')

st.markdown(
    """
    <style>
    .stProgress > div > div > div {
        box-shadow: 0 0 10px 0 rgba(0, 0, 0, 0.3) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Function to update the student list
def update_student_list():
    # Initialize the student data dictionary
    if 'student_data' not in st.session_state:
        st.session_state.student_data = {}

    # Get the number of students from user input
    num_students = st.number_input("Enter the number of students:", min_value=1, step=1,
                                   value=st.session_state.student_data.get('num_students', 1))

    # Initialize an empty list to store the grades
    student_grades = []

    # Loop to input grades for each student
    for i in range(num_students):
        default_grade = st.session_state.student_data.get(f"student_{i + 1}_grade", 0)
        grade = st.number_input(f"Enter the grade for student {i + 1}:", min_value=0, step=1, value=default_grade,
                                key=f"grade_{i}")
        student_grades.append(grade)

        # Update the student data dictionary
        st.session_state.student_data[f"student_{i + 1}_grade"] = grade

    # Update the student data dictionary with the number of students
    st.session_state.student_data['num_students'] = num_students

    # Convert the student grades list to a 2D array
    new_grades = np.array(student_grades).reshape(-1, 1)

    # Predict the difficulty level for the new chapter
    predicted_difficulty = model.predict(new_grades)

    # Calculate the mean difficulty level
    mean_difficulty = np.mean(predicted_difficulty)
    rounded_mean_difficulty = round(mean_difficulty)

    # Update the predicted difficulty levels with the new mapping
    difficulty_mapping = {
        0: "Weak",
        1: "Below Average",
        2: "Average",
        3: "Excellent"
    }
    predicted_difficulty = [difficulty_mapping.get(round(difficulty / 10), "Unknown") for difficulty in
                            predicted_difficulty]

    # Display the average performance of the class as text
    st.subheader("Average Performance of the Class")
    st.markdown(f"The average performance of the class is **{difficulty_mapping.get(rounded_mean_difficulty // 10, 'Unknown')}**.")

    # Display the mean difficulty level using a progress bar
    st.subheader("Progress Bar")
    st.progress(rounded_mean_difficulty / 30)  # Assuming the maximum difficulty level is 30

    # Return the mean difficulty level
    return rounded_mean_difficulty


def perform_key_term_extraction(question):
    # Perform key term extraction using NLTK
    tokens = word_tokenize(question)
    tagged_tokens = pos_tag(tokens)

    # Filter out key terms (NN: nouns, NNS: plural nouns, JJ: adjectives)
    key_terms = [token for token, tag in tagged_tokens if
                 (tag.startswith('NN') or tag == 'JJ') and token.lower() not in ['question', 'answer']]

    return key_terms


def search_websites(key_terms):
    # Join the key terms into a single search query
    query = ' '.join(key_terms)

    # Search for the query and get the URLs with a delay between requests
    urls = []
    for url in search(query):
        urls.append(url)
        time.sleep(5)  # Add a delay of 5 seconds between requests

        if len(urls) == 2:
            break

    return urls


def generate_questions(topic, grade, num_questions):
    prompt = f"This is a question paper generator for {grade}th grade on the topic of {topic}.\n\n"

    generated_questions = []

    for i in range(num_questions):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt + f"Question {i + 1}: ",
            temperature=0.6,
            max_tokens=100
        )

        question = response.choices[0].text.strip()
        generated_questions.append(f"Question {i + 1}: {question}")

        prompt += f"{question}\n\n"

    return generated_questions


def main():
    st.title("Welcome to Project Arindam 2.0!")

    choice = st.sidebar.selectbox("Select an option:", ["Update Student List", "Website Recommendation", "Generate Question Paper"])

    if choice == "Update Student List":
        update_student_list()

    elif choice == "Website Recommendation":
        st.subheader("Website Recommendation")
        question_file = st.file_uploader("Upload a file with questions", type=["txt"])

        if question_file is not None:
            questions = question_file.read().decode()
            questions_list = questions.split("\n")

            for question in questions_list:
                key_terms = perform_key_term_extraction(question)

                if key_terms:
                    st.markdown("### Key Terms")
                    st.write(", ".join(key_terms))
                    st.markdown("---")

                    urls = search_websites(key_terms)

                    if urls:
                        st.markdown("### Top Recommended Websites")
                        for i, url in enumerate(urls):
                            st.markdown(f"{i + 1}. [{url}]({url})")
                    else:
                        st.info("No websites found.")
                else:
                    st.info("No key terms extracted.")

                st.markdown("---")

        else:
            st.info("Please upload a text file with questions.")

    elif choice == "Generate Question Paper":
        st.subheader("Question Paper Generator")
        topic = st.text_input("Enter the topic:")
        grade = st.text_input("Enter the class:")
        num_questions = st.number_input("Enter the number of questions:", min_value=1, step=1)

        if st.button("Generate"):
            generated_questions = generate_questions(topic, grade, num_questions)

            if len(generated_questions) > 0:
                st.subheader("Generated Questions:")
                for i, question in enumerate(generated_questions):
                    st.write(question)
                    st.write("---")
            else:
                st.info("No questions generated.")

            if len(generated_questions) > 0:
                questions_text = "\n".join(generated_questions)
                b64_questions = base64.b64encode(questions_text.encode()).decode()
                download_button = f'<a href="data:file/txt;base64,{b64_questions}" download="generated_questions.txt"><button>Download Questions</button></a>'
                st.markdown(download_button, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("By Anshuman Mishra")


if __name__ == "__main__":
    main()
