import streamlit as st
from utils1 import *
from analysis1 import *


def student_analysis(grades_df, curriculum_df, elective_df, elective_curriculum_df, course_mapping):
    st.header("Student Grade Analysis")

    # Input roll number
    roll_no = st.text_input("Enter Roll Number:").upper()

    if roll_no:
        student_grades = get_student_grades(grades_df, roll_no)

        # Visualization options
        viz_type = st.selectbox(
            "Select Visualization",
            ["Grade Distribution", "Heatmap", "Performance Timeline"]
        )

        if viz_type == "Grade Distribution":
            # Create pie chart of student's grades with labels like "Grade A", "Grade B", etc.
            grade_counts = pd.Series(student_grades.values()).value_counts()

            # Add "Grade " prefix to each grade label
            grade_labels = [f"Grade {grade}" for grade in grade_counts.index]

            # Create the pie chart
            fig = px.pie(
                values=grade_counts.values,
                names=grade_labels,  # Use the modified labels here
                title=f"Grade Distribution for Roll No: {roll_no}",
                template="plotly_dark"
            )

            # Display the pie chart
            st.plotly_chart(fig, use_container_width=True)


        elif viz_type == "Heatmap":
            fig = create_heatmap(grades_df, roll_no, course_mapping)
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Performance Timeline":
            fig = create_performance_timeline(grades_df, roll_no, course_mapping)

            # Display the plot
            st.plotly_chart(fig, use_container_width=True, key="performance_timeline_chart")

def course_recommendations(grades_df, curriculum_df, elective_df, elective_curriculum_df, course_mapping):
    st.header("Course Recommendations")

    roll_no = st.text_input("Enter Roll Number:")

    if roll_no:
        student_grades = get_student_grades(grades_df, roll_no)
        best_courses = get_best_courses(student_grades)

        if best_courses:
            keywords = get_curriculum_keywords(curriculum_df, best_courses)
            similar_courses = find_similar_courses(elective_curriculum_df, keywords)

            st.subheader("Recommended Electives")
            for i, (_, course) in enumerate(similar_courses.iterrows()):
                with st.expander(f"📘 {course['Course Code']} - {course['Course Name']}"):
                    course_data = elective_df[elective_df['Course Code'] == course['Course Code']]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.plotly_chart(
                            create_hours_visualization(course_data),
                            use_container_width=True,
                            key=f"hours_viz_{i}"
                        )

                    with col2:
                        st.plotly_chart(
                            create_grades_distribution(course_data),
                            use_container_width=True,
                            key=f"grades_dist_{i}"
                        )

                    st.write(f"**Faculty Name:** {course_data['Faculty Name'].iloc[0]}")
                    st.write(f"**Credits:** {course_data['Credits'].iloc[0]}")


def general_statistics(grades_df, elective_df):
    st.header("General Course Statistics")

    # Hypothesis testing
    st.subheader("Hypothesis Testing")
    z_stat, p_value = perform_hypothesis_test(elective_df)
    st.write(f"Z-statistic: {z_stat:.2f}")
    st.write(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        st.write("The distribution of S and A grades is statistically different from the overall distribution.")
    else:
        st.write("The distribution of S and A grades is not statistically different from the overall distribution.")

    # PCA Analysis
    st.subheader("Course Clustering Analysis (PCA)")
    pca_fig = perform_pca_analysis(elective_df)
    st.plotly_chart(pca_fig, use_container_width=True)

    # Overall grade distribution
    st.subheader("Overall Grade Distribution")
    total_grades = {
        'S': elective_df['S'].sum(),
        'A': elective_df['A'].sum(),
        'B': elective_df['B'].sum(),
        'C': elective_df['C'].sum(),
        'D': elective_df['D'].sum(),
        'E': elective_df['E'].sum()
    }
    fig = px.pie(
        values=list(total_grades.values()),
        names=list(total_grades.keys()),
        title="Overall Grade Distribution",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    # Set page config
    st.set_page_config(
        page_title="Course Guru",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load and clean data
    grades_df, curriculum_df, elective_df, elective_curriculum_df = load_and_clean_data()
    course_mapping = get_course_mapping(elective_df)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Student Analysis", "Course Recommendations", "General Statistics"]
    )

    if page == "Student Analysis":
        student_analysis(grades_df, curriculum_df, elective_df, elective_curriculum_df, course_mapping)
    elif page == "Course Recommendations":
        course_recommendations(grades_df, curriculum_df, elective_df, elective_curriculum_df, course_mapping)
    elif page == "General Statistics":
        general_statistics(grades_df, elective_df)


if __name__ == "__main__":
    main()
