# football-data-analytics

This repository is about using various techniques to analyze event data of English Women's Super League (Season 18/19 - 20/21).
The projects are developed under WSL.

### Data source
The data is acquired from statsbomb and processed into my own MySQL database using Pyspark and SQL.

### Streamlit web app
The web app employs data visualization, statistical methods and machine learning to analyze event data. The web app includes the following components:
- Competition data
- Match data
- Team data
- Player data

### Jupyter notebook
##### [ETL processes (PySpark, Spark SQL, SQL queries)](https://github.com/MaggieFungly/football-data-analytics/blob/main/spark%20new.ipynb)
Acquires, processes and transform original statsbomb event data, and loads the transformed data into my own MySQL database.
#### Data analytics with machine learning techniques
##### [Identify goal scoring with event series](https://github.com/MaggieFungly/football-data-analytics/blob/main/goal_scoring_event_series_identification.ipynb)
- PySpark & Spark SQL: Extract data from MySQL database
- LightGBM: identifying goal-scoring with series events
- SHAP: valuing player actions before a shot is made
