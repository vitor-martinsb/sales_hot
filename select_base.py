import mysql.connector
import pandas as pd

# Connect to the MySQL database
cnx = mysql.connector.connect(
    host='localhost',
    user='your_username',
    password='your_password',
    database='your_database'
)

# Create a cursor object to execute SQL queries
cursor = cnx.cursor()

# Define the SQL query to retrieve data
query = 'SELECT * FROM your_table'

# Execute the query
cursor.execute(query)

# Fetch all rows returned by the query
rows = cursor.fetchall()

# Get column names from the cursor description
columns = [desc[0] for desc in cursor.description]

# Create a Pandas DataFrame from the fetched data
df = pd.DataFrame(rows, columns=columns)

# Process the DataFrame as needed
print(df.head())

# Close the cursor and database connection
cursor.close()
cnx.close()