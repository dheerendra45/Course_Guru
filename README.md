## Course Guru

**Course Guru** is a web-based platform designed to provide students with insights into their academic performance, recommend electives based on their strengths and curriculum, and showcase general statistics about courses and grades.

---

## Table of Contents
* [Features](#features)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Usage](#usage)
* [File Structure](#file-structure)
* [Future Enhancements](#future-enhancements)
  

---

## Features

### 1. **Student Analysis**

- Retrieve a student’s performance by roll number.
- Visualize:
  - Grade distribution (pie chart).
  - Grade heatmap (mapped to course codes).
  - Performance timeline (trend visualization).

### 2. **Course Recommendations**

- Recommends electives based on:
  - Student’s grades.
  - Curriculum keywords using TF-IDF and cosine similarity.
- Displays elective details such as:
  - Grade distributions.
  - Faculty information.
  - Hours required.

### 3. **General Statistics**

- Statistical insights into grade distributions.
- Principal Component Analysis (PCA) for clustering courses.
- Hypothesis testing for identifying patterns in grades.

---

## Technologies Used

- **Frontend**: [Streamlit]
- **Data Handling**: [Pandas]
- **Machine Learning**: [Scikit-learn]
  - TF-IDF Vectorization
  - Cosine Similarity
  - Principal Component Analysis (PCA)
- **Visualizations**: [Plotly]
- **Python Standard Libraries**: os, csv, numpy

---

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/course-guru.git
   cd course-guru
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```
5. Open your browser and navigate to the Streamlit local server (usually `http://localhost:8501`).

---

## Usage

1. Choose a section from the sidebar:
   - **Student Analysis**: Enter a roll number to analyze a student’s performance.
   - **Course Recommendations**: View recommended electives based on curriculum and grades.
   - **General Statistics**: Explore insights about overall grade distributions and course clusters.
2. Interact with visualizations and download reports as needed.

---

## File Structure

```
course-guru/
|-- app.py                 # Main Streamlit app file
|-- requirements.txt       # Python dependencies
|-- utils1.py              # Utility functions for student analysis
|-- analysis1.py           # Utility functions for course recommendations and statistics
|-- data/
|   |-- grades.csv         # Student grades data
|   |-- courses.csv        # Course information
|-- README.md              # Project documentation
```

---

## Future Enhancements

- Add user authentication for personalized dashboards.
- Include filters for elective recommendations (e.g., by department, level).
- Extend visualization options with advanced analytics.
- Integrate additional datasets for enriched recommendations.


**Developed with ❤️ by Course Guru Team**

