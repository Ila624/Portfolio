# Second-Hand Fashion Marketplace: User Behavior Analysis

Final data analysis project on a second-hand clothing marketplace, a platform that promotes sustainable fashion and the circular economy by letting users resell used clothes, accessories and shoes.

The goal is to explore sales data to understand seller behavior, identify successful products and designers, observe time trends, and propose a strategy for the coming year based on user personas.

## Project structure

The project is organized in two sequential phases:

1. **EDA in Python** (Jupyter Lab): dataset exploration, data quality checks, and analysis of products, designers, time trends and users.
2. **Interactive Tableau dashboard**: marketplace overview, segmentation of users into personas, and suggested strategic actions for each profile.

```
├── README.md
├── EDA_Fashion_Marketplace.ipynb          # full EDA
├── Fashion_data_Final_Project.csv         # original dataset (one row = one sale)
└── user_summary_tableau.csv               # per-user aggregated data, input for the dashboard
```

## Dataset

The dataset contains **999 sales** made by **241 users** across **23 designers**, over the period **1 January 2021 – 28 December 2023**.

| Column | Description |
|---|---|
| `purchase_date` | Date on which the user sold the product |
| `user_uuid` | Identifier of the selling user |
| `category` | Clothing category |
| `designer_id` | Designer identifier |
| `language` | Language of the selling user |
| `level` | Price tier (cheap, average, premium) |
| `country` | Country of the selling user |
| `subscription_date` | Date on which the user signed up |
| `platform` | Platform from which the payment was made |
| `item_id` | Product identifier |
| `stars` | Stars assigned to the product (1 to 5) |

## Exploratory data analysis (EDA)

The notebook follows these steps:

1. Setup and data loading
2. General overview and quality checks (data types, missing values, duplicates)
3. Date cleaning and feature engineering (year, month, weekday, account tenure at the time of sale)
4. Univariate analysis of categorical variables
5. Time trends (monthly and yearly trend, seasonality, day of the week)
6. Cross analysis: what drives product success (category, price tier, designer, ratings)
7. User-level analysis (selling frequency, prevailing price tier, tenure)
8. Summary of insights and conclusions

### Key findings

- **Strong growth** in sales volume from 2021 to 2023, a sign of an expanding marketplace. No particularly marked seasonal or weekly peaks emerge.
- **`belt` is the best-selling category** (339 sales, about 34%), an unexpected result for a fashion marketplace, where one would expect items such as shirts or shoes to dominate.
- **No relationship between sales volume and average rating** at designer level (correlation ≈ -0.24): with the available data, successful designers cannot be identified from star ratings.
- **User behavior**: 4.1 sales per user on average (median 4, range 1–11), with no extreme power sellers. About 23% of users made only 1–2 sales, while 7.5% made 8 or more.
- **Prevailing price tier per user**: premium 39%, cheap 32%, average 29%.

### Data quality notes

- **`item_id` is not a unique key**: 98 IDs are duplicated, but they are linked to items with different categories and/or designers. These are therefore ID collisions, not items resold multiple times.
- **`language` and `country` are redundant**: the mapping is strictly 1-to-1 with no exceptions (uk/en, it/it, fr/fr).
- **`stars` has an almost uniform distribution** (about 20% for each value from 1 to 5), an unusual pattern for real reviews that suggests synthetically generated data. For this reason `stars` was used with caution and not as the main basis for segmentation.

## Tableau dashboard

The dashboard turns the EDA results into an interactive tool to tell the story of user behavior and propose a strategy for the coming year. It is organized in three sections:

1. **Marketplace overview**: key figures and sales trends
2. **Personas**: segmentation of users into typical profiles, combining selling frequency and prevailing price tier
3. **Strategy**: suggested actions for each profile

🔗 [**Dashboard on Tableau Public**](https://public.tableau.com/views/MarketplaceSeconf-HandFascionProgettoFinaleB/IlMarketplaceinnumeri?:language=it-IT&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## Technologies

- Python 3: `pandas`, `numpy`, `matplotlib`, `seaborn`
- Jupyter Lab
- Tableau Public

## How to reproduce the analysis

1. Clone the repository
2. Install the dependencies: `pip install -r requirements.txt`
3. Open the notebook in Jupyter Lab and, from the Kernel menu, choose "Restart Kernel and Run All Cells". Cells must be run in order from the top, because some depend on variables created in earlier ones.
4. The last cell of the notebook exports `user_summary_tableau.csv`, which is used as the data source in Tableau.

## Presentation

🔗 [**Presentazione progetto (IT)**](https://canva.link/2beho7ltrpz99mh)
🔗 [**Project presentation (EN)**](#)
