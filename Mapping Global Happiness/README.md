# Mapping Global Happiness: A Five-Year Data Analysis (2015–2019)


An end-to-end data analytics project exploring the macroeconomic, social, and cultural drivers of world happiness prior to the COVID-19 pandemic. This project leverages Python for data cleaning and integration, and Tableau for interactive data visualization and storytelling.

 🎨 **[View the Presentation Slides on Canva](YOUR_CANVA_SLIDES_LINK)** <br>🔗 **[View the Interactive Tableau Dashboard](YOUR_TABLEAU_PUBLIC_DASHBOARD_LINK)**

---

## 📌 Project Overview
This project analyzes global happiness trends using the **Gallup World Poll** datasets from 2015 to 2019 inclusive. The core metric, the **Happiness Score** (evaluated via the Cantril Ladder scale), aggregates survey responses from up to 157 countries to measure national life satisfaction. 

The primary goal of this analysis is to go **"Beyond GDP"**—investigating how economic indicators interact with immaterial factors (such as Social Support, Freedom, and Institutional Trust) to drive overall well-being.

---

## 🛠️ Data Pipeline & Technical Challenges
The raw data was initially distributed across five separate `.csv` files, featuring inconsistent column names, varying dimensions, and missing records for certain years.

To ensure data integrity prior to visualization, a **Data Preparation pipeline** was built in Python using **Pandas**:
* **Standardization:** Column headers were unified across all 5 datasets.
* **Feature Engineering:** A dedicated `Year` column was generated to enable time-series analysis.
* **Dimensionality Reduction:** Irrelevant or non-shared fields were omitted to yield a clean, unified dataset.
* **Data Quality Note:** Countries with missing data for specific years are dynamically rendered in gray within the dashboard map to maintain statistical honesty.

---

## 📊 Key Insights & Analytical Findings
* **The "Wealth Ceiling" (Diminishing Marginal Returns on GDP):** While economic wealth is a fundamental prerequisite for happiness (particularly in developing macro-regions), past a certain threshold GDP growth stops correlating with higher well-being. The comparison between Northern Europe (high happiness/wealth) and Hong Kong (average happiness/extreme wealth) underscores that life satisfaction depends on transforming wealth into public welfare.
* **Health as a Driver:** Health Expectancy emerges as a highly impactful factor with very low data dispersion. Europe shows the steepest regression slope (small health improvements lead to major happiness gains), whereas Africa displays a flatter trajectory.
* **Social Safety Nets as Shock Absorbers:** Deep dives into the Middle East and Latin America (e.g., Costa Rica as a prominent socio-economic outlier) demonstrate that community support networks and personal freedom act as vital emotional shock absorbers during national crises.

---

## 🎛️ Dashboard Structure & Interactivity
The interactive Tableau dashboard is engineered around 4 main visualizations and advanced interactive elements:
* **Interactive World Map:** Features dynamic tooltips and acts as a global dashboard filter upon selecting a nation.
* **Factor Correlation Scatter Plot:** Uses a **Parameter Control** allowing users to dynamically swap the independent variable ($x$-axis) among 6 key metrics. Includes regional trend lines.
* **Happiness Trend Over Time:** Tracks historical trajectories benchmarked against a global average reference line.
* **Collapsible Details Pane:** A hidden sheet triggered via a "Country Details" button, providing a granular, horizontal bar chart analysis of a single country's active factors.

---

---

## 📑 Data Sources & References

The analysis and methodology of this project are strictly grounded in official statistical frameworks and international reports. Detailed documentation and raw reference data can be accessed through the following sources:

* **[World Happiness Report Data Repository](https://worldhappiness.report/ed/2019/#appendices-and-data):** The official public datasets and statistical appendices tracking global life satisfaction scores from 2015 to 2019.
* **[Gallup World Poll](https://www.gallup.com/analytics/318875/global-research.aspx ):** The primary authority responsible for the comprehensive global surveys, field research, and data collection methodology.
* **[How Gallup uses the Cantril Scale](https://news.gallup.com/poll/122453/understanding-gallup-uses-cantril-scale.aspx):** Gallup's official guide explaining the self-anchoring striving scale used during interviews to evaluate perceived well-being.
