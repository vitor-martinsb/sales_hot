Client Segmentation and Feature Analysis
This repository contains code for client segmentation and feature analysis based on RFM (Recency, Frequency, Monetary) analysis.

Table of Contents
Introduction
Installation
Usage
Example
Contributing
License
Introduction
The code in this repository performs client segmentation and feature analysis using the RFM analysis technique. It reads data from either a CSV file or a SQL table, calculates and plots the valuation and faturation of products, performs RFM analysis to segment clients, and performs feature analysis to select relevant features and plot feature importance.

Installation
To run the code in this repository, you need to have the following dependencies installed:

Python (version 3.6 or higher)
pandas
tqdm
matplotlib
scikit-learn
You can install the dependencies by running the following command:

bash
Copy code
pip install -r requirements.txt
Usage
To use the code, follow these steps:

Clone this repository to your local machine.

Open a terminal or command prompt and navigate to the cloned repository's directory.

Run the code using the following command:

bash
Copy code
python main.py
Note: Make sure you have the required data files in the appropriate directories (data/sales_data.csv, data/RFM.csv) or modify the code accordingly to read the data from your desired location.

Example
Here's an example of how to use the code:

python
Copy code
from main import read_data

rd = read_data()
# Perform valuation and faturation analysis
df_product_category_value, df_product_niche_value = rd.val_faturation(media=100, desvio_padrao=10)

# Perform client segmentation
rd.seg_client(media=100, desvio_padrao=10, read_RFM_data=True)

# Perform feature analysis
rd.feature_analysis()
Contributing
Contributions to this repository are welcome. If you have any suggestions, improvements, or bug fixes, please submit a pull request.

License
This project is licensed under the MIT License.

