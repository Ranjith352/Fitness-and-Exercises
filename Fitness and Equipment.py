# Required Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import spacy
import streamlit as st

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Function to load dataset
def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        data['Category'] = data['Category'].str.strip().str.lower()  # Normalize category names
        return data
    except FileNotFoundError:
        st.error("Error: The dataset file was not found.")
        st.stop()

# Function to enhance queries with keywords
def map_query_to_category(query):
    keywords = {
        'legs': 'legs exercises',
        'leg': 'legs exercises',
        'arm': 'arms exercises',
        'arms': 'arms exercises',
        'abdominal': 'core strengthening',
        'core': 'core strengthening',
        'upper body': 'upper body exercises',
        'chest': 'chest exercises',
        'back': 'back exercises',
        'shoulder': 'shoulder exercises'
    }
    query_lower = query.lower()
    for keyword, category in keywords.items():
        if keyword in query_lower:
            return category
    return None  # Return None if no keywords matched

# Function to extract excluded categories from user input
def get_excluded_categories(query):
    doc = nlp(query)
    excluded_categories = []
    
    # Identify negations
    for token in doc:
        if token.dep_ == 'neg':
            # Collect the exercise or category associated with the negation
            if token.head.text in ['need', 'want']:
                for child in token.head.children:
                    if child.dep_ == 'dobj':
                        excluded_category = map_query_to_category(child.text)
                        if excluded_category:
                            excluded_categories.append(excluded_category)
    
    return excluded_categories

# Function to perform text classification
def text_classification(data):
    # Create a pipeline for text classification
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['Exercise'], data['Category'], test_size=0.2, random_state=42)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    return model

# Home Page
def home_page():
    st.title("Welcome to Gym and Equipment Management")
    st.image("C:\\Users\\ranja\\OneDrive\\Desktop\\Ranjith\\Semester 5\\NLP Lab\\Project\\Intro.png", 
             caption="Fitness and Exercises", use_column_width=True)
    st.write("""Fitness involves physical activities that improve health and wellness. 
        It is essential for maintaining a healthy body and mind. 
        Here are some aspects of fitness:
        - *Cardio:* Involves activities that raise your heart rate, such as running and cycling.
        - *Strength Training:* Focuses on building muscle through resistance exercises.
        - *Flexibility:* Enhances the range of motion in joints through stretching.
        - *Balance:* Important for preventing falls and injuries, especially in older adults.""")

# Fitness and Exercises Page
def fitness_exercises_page():

    st.write("### Recommended Exercise GIFs:")

    st.subheader("Shoulder Exercises")
    st.image(r"C:\Users\ranja\OneDrive\Desktop\Ranjith\Semester 5\NLP Lab\Project\Shoulder.gif", caption="Shoulder Exercises", use_column_width=True)
    st.write("""
        Shoulder exercises are crucial for maintaining shoulder stability and mobility. 
        They help strengthen the rotator cuff and can reduce the risk of injury. 
        Incorporating shoulder exercises into your routine improves overall upper body strength and enhances performance in sports and daily activities.
    """)
    
    st.subheader("Leg Exercises")
    st.image(r"C:\Users\ranja\OneDrive\Desktop\Ranjith\Semester 5\NLP Lab\Project\Legs.gif", caption="Leg Exercises", use_column_width=True)
    st.write("""
        Leg exercises are vital for building strength in your lower body, including the quadriceps, hamstrings, calves, and glutes. 
        Strong legs enhance athletic performance, support mobility, and improve balance. 
        Regular leg workouts can also aid in weight management and overall fitness levels.
    """)
    
    st.subheader("Core Strengthening Exercises")
    st.image(r"C:\Users\ranja\OneDrive\Desktop\Ranjith\Semester 5\NLP Lab\Project\Core Strengthening.gif", caption="Core Strengthening Exercises", use_column_width=True)
    st.write("""
        Core strengthening exercises are essential for improving stability, balance, and posture. 
        A strong core supports daily activities and enhances athletic performance. 
        It also reduces the risk of injuries by providing a solid foundation for movements involving the arms and legs.
    """)
    
    st.subheader("Back Exercises")
    st.image(r"C:\Users\ranja\OneDrive\Desktop\Ranjith\Semester 5\NLP Lab\Project\Back.gif", caption="Back Exercises", use_column_width=True)
    st.write("""
        Back exercises help strengthen the muscles in your back, which is crucial for maintaining good posture and preventing back pain. 
        These exercises enhance your functional strength, making everyday activities easier. 
        Additionally, a strong back contributes to better balance and stability during physical activities.
    """)
    
    st.subheader("Arm Exercises")
    st.image(r"C:\Users\ranja\OneDrive\Desktop\Ranjith\Semester 5\NLP Lab\Project\Arm.gif", caption="Arm Exercises", use_column_width=True)
    st.write("""
        Arm exercises are key to building strength in the biceps, triceps, and shoulders. 
        Strong arms improve performance in various sports and daily tasks that require pushing or pulling. 
        Regular arm workouts can enhance muscle definition and overall upper body strength.
    """)


# User Input Page
def user_input_page():
    st.title("User Details Input")
    
    name = st.text_input("Name:", key="user_name")
    age = st.number_input("Age:", min_value=1, max_value=100, key="user_age")
    gender = st.selectbox("Gender:", ["Male", "Female", "Other"], key="user_gender")
    location = st.text_input("Location:", key="user_location")
    
    if st.button("Submit", key="submit_user_details"):
        st.success(f"Details submitted for {name}!")

# Query Result Page
def query_page(data, model):
    st.title("Query Input")
    
    user_query = st.text_input("Enter your informal query:", "", key="query_input")
    
    if user_query:
        # Check for direct keyword mapping
        predicted_category = map_query_to_category(user_query)

        # Initialize variable for displaying exercises
        available_exercises = {}

        # If a predicted category is found, process it
        if predicted_category:
            # Create Knowledge Base
            knowledge_base = {
                row['Exercise']: {
                    'Category': row['Category'],
                    'Location': row['Location'],
                    'Sets': row['Sets'],
                    'Reps': row['Reps'],
                    'Equipment': row['Equipment']
                } 
                for idx, row in data.iterrows()
            }

            # Get excluded categories from user query
            excluded_categories = get_excluded_categories(user_query)

            if "do not" in user_query or "don't" in user_query:
                # User does not want exercises of the predicted category
                available_exercises = {
                    exercise: details for exercise, details in knowledge_base.items()
                    if details['Category'] != predicted_category and details['Category'] not in excluded_categories
                }
                # Display Results
                if available_exercises:
                    st.write(f"### Exercises Excluding {predicted_category}:")
                    for exercise, details in available_exercises.items():
                        st.write(f"- **Exercise:** {exercise}")
                        st.write(f"  - *Sets:* {details['Sets']}")
                        st.write(f"  - *Reps:* {details['Reps']}")
                        st.write(f"  - *Equipment:* {details['Equipment']}")
                else:
                    st.write("No exercises found outside the excluded categories.")
            else:
                # User wants exercises of the predicted category
                available_exercises = {
                    exercise: details for exercise, details in knowledge_base.items()
                    if details['Category'] == predicted_category
                }
                # Display Results
                if available_exercises:
                    st.write(f"### Exercises Related to {predicted_category}:")
                    for exercise, details in available_exercises.items():
                        st.write(f"- **Exercise:** {exercise}")
                        st.write(f"  - *Sets:* {details['Sets']}")
                        st.write(f"  - *Reps:* {details['Reps']}")
                        st.write(f"  - *Equipment:* {details['Equipment']}")
                else:
                    st.write(f"No exercises found for {predicted_category}.")
        else:
            st.write("Could not identify any exercise category from your query.")

# Feedback Page
def feedback_page():
    st.title("Feedback")
    feedback = st.text_area("Please provide your feedback:", key="user_feedback")
    
    if st.button("Submit Feedback", key="submit_feedback"):
        st.success("Thank you for your feedback!")

# Text Analysis Page
def text_analysis_page(data, model):
    st.title("Text Analysis")

    user_query = st.text_input("Enter your informal query for analysis:", "", key="text_analysis_input")
    
    if user_query:
        # Syntax and Semantic Analysis
        doc = nlp(user_query)

        # Syntax Analysis
        st.write("### Syntax Analysis:")
        syntax_analysis = [(token.text, token.pos_) for token in doc]
        st.write(syntax_analysis)

        # Perform text classification
        predicted_category = model.predict([user_query])[0]
        st.write(f"### Predicted Exercise Category: {predicted_category}")

        # Semantic Analysis
        st.write("### Semantic Analysis:")
        for token in doc:
            st.write(f"{token.text}: {token.dep_}, Head: {token.head.text}")

        # Chunking (Noun Phrases)
        st.write("### Noun Phrases:")
        chunks = [(chunk.text, chunk.root.head.text) for chunk in doc.noun_chunks]
        st.write(chunks)

# Main Function to Manage Navigation
def main():
    # Load dataset
    data_filepath = "C:\\Users\\ranja\\OneDrive\\Desktop\\Ranjith\\Semester 5\\NLP Lab\\Knowledge_base_dataset.csv"
    data = load_data(data_filepath)

    # Train the model
    model = text_classification(data)

    st.sidebar.title("Navigation")
    pages = {
        "Home": home_page,
        "Fitness and Exercises": fitness_exercises_page,
        "User Input": user_input_page,
        "Query": lambda: query_page(data, model),  # Pass arguments properly
        "Feedback": feedback_page,
        "Text Analysis": lambda: text_analysis_page(data, model),  # Pass arguments properly
    }
    
    # Sidebar selection
    page_selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Call the selected page function
    pages[page_selection]()

# Run the app
if __name__ == "__main__":
    main()
