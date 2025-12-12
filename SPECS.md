# SCM Lab

The purpose of this project is to build a general-purpose tool that allows researchers to create synthetic control models (SCMs) to evaluate the impact of a one-time event on a single treated unit by creating a synthetic control unit. This control is a weighted average of similar, untreated units that mimic the treated unit's behavior before the intervention. To create a control, the program will use a regression model whose weights are meant to converge to a combination that minimizes the mean squared error (MSE) of the predictions of the pre-intervention treated unit.

## Data Source

The data should be sourced from reputable research projects focused on calculating key macroeconomic metrics like GDP per capita over time.

The first data source for this project will be the Maddison Project’s estimations of the GDP per capita of various nations from 1 to 2022 CE. The Maddison Project’s data format is a CSV where the first row contains the headers, the subsequent rows contain the data, and the columns are as follows:
- Entity (country, e.g., “Italy”)
- Code (3-letter country code, e.g., “ITA”)
- Year (year, e.g., 2022)
- GDPPerCapita (GDP per capita, e.g., 36224.293)
- Annotations (unimportant column)

## Training Data

The regression model receives training data in the following format for a given range of years that make up the pre-treatment period:
- Each year in the pre-treatment period has its own row.
- In each row, there is one input column for each untreated country and one output column for the treated country.

## Program Flow

The user should be able to decide whether to view a saved SCM or create a new SCM.

If the user decides to view a saved SCM, then:
- The user should be able to choose between all their saved SCMs.
- Once the user chooses the saved SCM, the program will load up a table consisting of the following data for the user to see:
    - One input column for each untreated country.
    - One factual output column for the treated country showing the actual metrics for that country the time period in focus.
    - One synthetic output column for the treated country showing the estimated metrics for that country the time period in focus.
- There should also be a line graph above the table representing the data from the table as follows:
    - The x-axis is the year.
    - The y-axis is the metric (e.g., GDP per capita).
    - Each column has its own line in the graph.

If the user decides to create a new SCM, then:
- The user should be asked to decide the following:
    - A data source
    - A start year
    - An end year
    - A treatment year
    - A set of input countries
    - A single output country
- The program will build an appropriate training data table for training the regression model, as described in the section about training data.
    - The training data should consist of the metrics for each input country and the single output country from the start year to the year immediately before the treatment year.
- The linear regression model will be trained using the training data and output synthetic values for every year from the start year to the end year.
- A file consisting of the following data should be saved to a database: the metric for each input country, the single factual output country, and the single synthetic output country for every year from the start year to the end year.
- The program should display the data stored to this file as a table and above that a line graph based on the stored data in the same format as described earlier in the scenario where the user views a saved SCM.

## Files, Languages, and Libraries

The ultimate data sources will be CSVs.
- The Maddison Project data on GDP per capita will come in a file called gdp-per-capita-maddison.csv, though the files for future data sources will have different names.

Use Python to implement the backend in a file called main.py.

In main.py, use NumPy to store and manipulate data in memory, and to program the training of the regression model.

Use an HTML file called index.html, a CSS called styles.css, and a JavaScript file called script.js to implement the frontend.

Use a Python FastAPI file called app.py to enable API communication between the frontend and backend.

Use a SQL database file called data.db to create the database for storing data that can be loaded again after the program stops and restarts.

Include a requirements.txt listing all the essential imports.

Include a README.md explaining this project and providing credit to all relevant sources, including the [Maddison Project Database 2023](https://www.rug.nl/ggdc/historicaldevelopment/maddison/releases/maddison-project-database-2023) along with its authors Jutta Bolt and Jan Luiten van Zanden.
