Markdown
# US Hospitality Sector: Sentiment & Regional Grievance Analysis

This project delivers a data-driven evaluation of the US hospitality sector using Natural Language Processing (NLP) and geospatial analysis. By extracting customer pain points from hotel reviews, the analysis maps regional satisfaction trends and proposes strategic recommendations for service optimization and infrastructure modernizations.

## 📊 Core Features
* **Geospatial Mapping:** Interactive nationwide visualization of hotel clusters and performance metrics using `Folium` and `GeoPandas`.
* **Text Mining & Topic Modeling:** Unsupervised Learning via Latent Dirichlet Allocation (LDA) to group negative feedback into four core operational areas.
* **Regional Demographics:** Statistical breakdown of guest grievances across the 4 US Census macro-regions.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git)
   cd YOUR-REPOSITORY-NAME
Dataset Setup (Important):
The original dataset Datafiniti_Hotel_Reviews.csv is too large to be hosted directly on GitHub.
Please download the dataset from https://www.kaggle.com/datasets/datafiniti/hotel-reviews?select=Datafiniti_Hotel_Reviews.csv and place the CSV file in the root directory of this project before running the notebook.

Install dependencies:
Make sure you are using a Python environment with NumPy 1.x compatibility:

Bash
pip install -r requirements.txt
Run the Notebook:
Once the dataset is in place, launch Jupyter and run the analysis:

Bash
jupyter notebook HotelReviews(EN).ipynb
