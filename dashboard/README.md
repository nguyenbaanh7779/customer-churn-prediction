# Retention targeting dashboard

Run from the repo root (after notebooks 10-12 and `pip install -r requirements.txt`):

    streamlit run dashboard/app.py

Needs `data/results/customer_scores_actions_h90.parquet` and `data/processed/modeling_dataset_h90.parquet`. No model is retrained.

**KPIs (current list, scored Monday 5 Dec 2011):** active customers; revenue at risk in the offer list; expected incremental margin; offers to likely buyers; customers per action (Retain / Nurture / No-offer).

**Actionable view:** sidebar controls (discount, response, redemption, margin, contact cost, selection rule, discount budget or contact share) update the target list (downloadable CSV) and a chart comparing your policy with no offer, offer to everyone and longest-since-purchase on the Jul-Sep 2011 test weeks.
