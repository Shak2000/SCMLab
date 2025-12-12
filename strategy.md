# SCM Lab Development Strategy

This document outlines the development strategy for the SCM Lab project. The project is divided into several phases, each with specific milestones, objectives, and tests to ensure a structured and robust development process.

---

## Phase 1: Project Setup and Backend Foundation

This phase focuses on establishing the project structure and implementing the core data handling capabilities of the backend.

### Milestone 1.1: Environment and File Structure
- **Objective:** Initialize the project directory with all necessary files and set up the Python environment.
- **Tasks:**
    - Create the file structure: `main.py`, `app.py`, `index.html`, `styles.css`, `script.js`, `requirements.txt`, `README.md`, and an empty `data.db`.
    - Create a Python virtual environment.
    - Install initial dependencies: `numpy`, `fastapi`, `uvicorn`, `SQLAlchemy`, and `pandas` (for CSV reading).
    - Populate `requirements.txt` with the installed libraries.
- **Tests:**
    - Verify that all files are created in the correct locations.
    - Confirm that `pip install -r requirements.txt` runs successfully in a clean environment.

### Milestone 1.2: Data Loading and Processing
- **Objective:** Implement the logic to load and prepare the dataset for the regression model.
- **Tasks:**
    - In `main.py`, create a function to load the `gdp-per-capita-maddison.csv` file into a suitable data structure (e.g., a Pandas DataFrame).
    - Implement a function to process the raw data into the specified training format using NumPy arrays. This function will take a start year, an end year, a treatment year, a list of untreated countries, and one treated country as input.
- **Tests:**
    - Write a unit test to ensure the CSV is loaded correctly.
    - Write a unit test to verify that the data processing function returns a correctly shaped NumPy array for the training data (features and target).

---

## Phase 2: Core SCM Logic Implementation

This phase involves building the regression model and the logic for generating the synthetic control.

### Milestone 2.1: Regression Model Training
- **Objective:** Develop the regression model that finds the optimal weights for the control units.
- **Tasks:**
    - In `main.py`, implement a function that takes the training data (from Milestone 1.2) and trains a linear regression model.
    - The model should be optimized to minimize the Mean Squared Error (MSE) between the actual and predicted pre-treatment outcomes for the treated unit.
- **Tests:**
    - Create a unit test with a known, simple dataset to verify that the model calculates the correct weights.

### Milestone 2.2: Synthetic Data Generation
- **Objective:** Use the trained model to generate the synthetic control data for the entire time period.
- **Tasks:**
    - In `main.py`, create a function that uses the calculated weights to generate the synthetic version of the treated unit's data from the specified start year to the end year.
- **Tests:**
    - Write a unit test to ensure the synthetic data is generated correctly based on a predefined set of weights and input data.

---

## Phase 3: API and Database Integration

This phase focuses on exposing the backend logic through an API and setting up the database for persistence.

### Milestone 3.1: FastAPI Endpoints
- **Objective:** Create API endpoints to handle requests from the frontend.
- **Tasks:**
    - In `app.py`, set up the FastAPI application.
    - Create a `/create_scm` POST endpoint that accepts the SCM parameters (years, countries, etc.), calls the backend logic from `main.py`, and returns the complete dataset (factuals and synthetic).
    - Create a `/scms` GET endpoint to list all saved SCMs.
    - Create a `/scms/{scm_id}` GET endpoint to retrieve the data for a specific saved SCM.
- **Tests:**
    - Use FastAPI's auto-generated documentation to manually test the endpoints with sample inputs.
    - Write automated API tests to verify endpoint responses and status codes.

### Milestone 3.2: Database Persistence
- **Objective:** Implement the database schema and logic for saving and retrieving SCM results.
- **Tasks:**
    - In `app.py` (or a dedicated database module), define the SQL schema for storing SCM results in `data.db`. This will include the parameters and the resulting time-series data.
    - Modify the `/create_scm` endpoint to save the newly generated SCM to the database.
    - Implement the logic for the `/scms` and `/scms/{scm_id}` endpoints to fetch data from the database.
- **Tests:**
    - Write tests to confirm that creating an SCM correctly saves the data to the database.
    - Write tests to confirm that the GET endpoints correctly retrieve the saved data.

---

## Phase 4: Frontend Development

This phase involves building the user interface for creating and viewing SCMs.

### Milestone 4.1: UI for SCM Creation
- **Objective:** Build the form that allows users to define the parameters for a new SCM.
- **Tasks:**
    - In `index.html`, create the main page layout with two primary options: "Create New SCM" and "View Saved SCMs".
    - Design the input form with fields for start/end/treatment years and selectors for the countries.
    - In `script.js`, write the JavaScript to handle form submission, collect the user inputs, and send them to the `/create_scm` API endpoint.
- **Tests:**
    - Manually test the form in a web browser to ensure all fields work as expected.
    - Verify that the correct request is sent to the backend upon submission.

### Milestone 4.2: Results Visualization
- **Objective:** Create the UI components for displaying the SCM results as a table and a line graph.
- **Tasks:**
    - In `index.html`, create container elements for the graph and the data table.
    - Use a JavaScript charting library (e.g., Chart.js) to implement the line graph, which will visualize the factual and synthetic data over time.
    - In `script.js`, write the logic to parse the API response and dynamically populate both the chart and the data table.
- **Tests:**
    - Manually test the visualization with sample data from the API to ensure the chart and table render correctly.

### Milestone 4.3: UI for Viewing Saved SCMs
- **Objective:** Implement the user flow for selecting and viewing a previously saved SCM.
- **Tasks:**
    - In `script.js`, implement the logic to call the `/scms` endpoint and display a list of saved SCMs to the user.
    - When a user selects a saved SCM, call the `/scms/{scm_id}` endpoint and use the returned data to populate the results view (chart and table from Milestone 4.2).
- **Tests:**
    - Manually test the complete "view saved" workflow to ensure it fetches and displays data correctly.

---

## Phase 5: Finalization and Documentation

This final phase focuses on polishing the application, adding documentation, and preparing for release.

### Milestone 5.1: Refinement and Error Handling
- **Objective:** Improve the user experience and make the application more robust.
- **Tasks:**
    - Enhance the UI/UX with improved styling in `styles.css`.
    - Implement frontend error handling (e.g., display messages for failed API requests).
    - Add loading indicators to provide feedback during long operations like model training.
- **Tests:**
    - Conduct end-to-end testing of the full application flow.
    - Test for edge cases, such as invalid user inputs or empty datasets.

### Milestone 5.2: Project Documentation
- **Objective:** Provide clear and comprehensive documentation for the project.
- **Tasks:**
    - Complete the `README.md` file with a project description, setup instructions, and usage guide.
    - Ensure proper credit is given to the Maddison Project Database and its authors in the `README.md`.
    - Review all code for clarity and add comments where the logic is complex.
- **Tests:**
    - Proofread all documentation for clarity, accuracy, and completeness.
    - Ask a peer to follow the `README.md` to set up and run the project from scratch.
