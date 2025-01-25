import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import plotly.express as px


def load_and_clean_data():
    """Load and clean all datasets"""
    grades_df = pd.read_csv('grades.csv')
    curriculum_df = pd.read_csv('curriculum.csv')
    elective_df = pd.read_csv('elective.csv')
    elective_curriculum_df = pd.read_csv('elective_curriculum.csv')

    # Clean grades dataframe
    grades_df = grades_df.dropna(subset=['Roll No'])
    return grades_df, curriculum_df, elective_df, elective_curriculum_df

def get_course_mapping(elective_df):
    """Load course code to name mapping"""
    return course_mapping

def get_student_grades(grades_df, roll_no):
    """Get grades for a specific student"""
    student_grades = grades_df[grades_df['Roll No'] == roll_no].iloc[0]
    return {col: grade for col, grade in student_grades.items() if col != 'Roll No' and pd.notna(grade)}

def get_best_courses(student_grades, threshold_grades=['S']):
    """Get courses where student performed well"""
    return [course for course, grade in student_grades.items() if grade in threshold_grades]

def get_curriculum_keywords(curriculum_df, course_codes):
    """Get curriculum keywords for given courses"""
    return curriculum_df[curriculum_df['Course Code'].isin(course_codes)]['Curriculum Keywords']


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def find_similar_courses(elective_curriculum_df, keywords, top_n=5):
    """Find similar courses using TF-IDF and cosine similarity"""
    # Convert keywords to a list if it's a single string or a Series
    if isinstance(keywords, pd.Series):
        keywords = keywords.tolist()  # Convert Series to list
    elif isinstance(keywords, str):
        keywords = [keywords]  # Wrap single string in a list
    elif not keywords or not any(keywords):  # Check if keywords list is empty or contains only empty strings
        return elective_curriculum_df.iloc[[9,10,11,14]]  # Return the 5th course if keywords is empty or invalid

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(elective_curriculum_df['Curriculum Keywords'])
        keyword_vector = vectorizer.transform(keywords)
        similarities = cosine_similarity(keyword_vector, tfidf_matrix)

        # Get top N indices for similar courses
        similar_indices = similarities.argsort()[0][-top_n:][::-1]
        return elective_curriculum_df.iloc[similar_indices]

    except ValueError:
        # Return the 5th course if there is an error (e.g., keywords is empty after transformation)
        return elective_curriculum_df.iloc[[9,10,11]]


def create_hours_visualization(course_data):
    """Create visualization for course hours distribution"""
    hours_data = {
        'Type': ['Lecture', 'Tutorial', 'Practical'],
        'Hours': [course_data['Lecture'].iloc[0], course_data['Tutorial'].iloc[0], course_data['Practical'].iloc[0]]
    }
    fig = px.bar(hours_data, x='Type', y='Hours', title='Course Hours Distribution', color='Type', template='plotly_dark')
    return fig

def create_grades_distribution(course_data):
    """Create advanced visualization for grades distribution with custom labels and colors"""
    # Define data with full grade labels
    grades_data = {
        'Grade': ['Grade S', 'Grade A', 'Grade B', 'Grade C', 'Grade D', 'Grade E'],
        'Count': [
            course_data['S'].iloc[0],
            course_data['A'].iloc[0],
            course_data['B'].iloc[0],
            course_data['C'].iloc[0],
            course_data['D'].iloc[0],
            course_data['E'].iloc[0]
        ]
    }

    # Define custom colors for each grade
    colors = {
        'Grade S': '#4CAF50',  # green
        'Grade A': '#2196F3',  # blue
        'Grade B': '#FFC107',  # amber
        'Grade C': '#FF5722',  # orange
        'Grade D': '#9C27B0',  # purple
        'Grade E': '#F44336'   # red
    }

    # Create pie chart with custom colors and labels
    fig = px.pie(
        grades_data,
        values='Count',
        names='Grade',
        title='Past Grade Distribution',
        template='plotly_dark',
        color='Grade',  # Apply colors based on the Grade label
        color_discrete_map=colors  # Use custom color mapping
    )

    # Update traces for display options
    fig.update_traces(
        textinfo='label+percent',  # Show label and percentage
        hoverinfo='label+percent+value',  # Show detailed info on hover
        textfont_size=14
    )

    # Update layout for title and legend
    fig.update_layout(
        title=dict(
            text='Past Grade Distribution',
            font=dict(size=20)
        ),
        showlegend=True
    )

    return fig


def create_heatmap(grades_df, roll_no, course_mapping):
    """Create an attractive heatmap of grades for a specific student, showing course codes and names."""
    grade_values = {'S': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'E': 0}
    student_grades = get_student_grades(grades_df, roll_no)

    # Fix the issue with student grades DataFrame
    numeric_grades = pd.DataFrame(student_grades, index=[roll_no])
    numeric_grades = numeric_grades.replace(grade_values)
    course_codes = numeric_grades.columns

    # Create course labels by fetching course names from the course mapping
    course_labels = [f"{course_code[:6]} {course_mapping.get(course_code[:6], '')}" for course_code in course_codes]


    # Reshape `numeric_grades` data for compatibility with `imshow`
    grades_matrix = numeric_grades.values.reshape(1, -1)

    # Create the heatmap
    fig = px.imshow(
        grades_matrix,
        x=course_labels,
        y=["Grade Level"],
        color_continuous_scale='plasma',  # Warm color scale
        title=f'Grade Distribution Heatmap for Roll No: {roll_no}'
    )

    # Update layout for better visuals
    fig.update_layout(
        xaxis_title="Courses (Code - Name)",
        yaxis_title="",
        template='plotly_dark',
        coloraxis_colorbar=dict(
            title="Grade Value",
            tickvals=[0, 1, 2, 3, 4, 5],
            ticktext=["Grade E", "Grade D", "Grade C", "Grade B", "Grade A", "Grade S"],
            len=0.8
        ),

    )

    # Adjust x-axis label angle for readability
    fig.update_xaxes(tickangle=45, tickmode='array', showgrid=True)

    return fig


# Import course mapping
from course_mapping import course_mapping

import plotly.express as px
import pandas as pd


def create_performance_timeline(grades_df, roll_no, course_mapping):
    """Create performance timeline visualization"""


    student_grades = grades_df[grades_df['Roll No'] == roll_no].dropna(axis=1, how='all').iloc[0]


    course_codes = student_grades.index[1:]
    grades = student_grades.values[1:]


    course_names = [course_mapping.get(course[:6], course[:6]) for course in course_codes]


    timeline_data = pd.DataFrame({
        'Course Code': course_codes,
        'Course Name': course_names,
        'Grade': grades
    })


    grade_order = ["E", "D", "C", "B", "A", "S"]


    timeline_data['Grade'] = pd.Categorical(timeline_data['Grade'], categories=grade_order, ordered=True)



    fig = px.line(
        timeline_data,
        x='Course Name',
        y='Grade',
        title=f"Grade Timeline for Roll No: {roll_no}",
        template="plotly_dark"
    )


    fig.update_layout(
        xaxis_title="Course",
        yaxis_title="Grade",
        yaxis_type="category",
        yaxis_categoryorder="array",
        yaxis_categoryarray=grade_order,
        xaxis_tickangle=45
    )

    # Return the figure
    return fig

