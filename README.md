

**Author:** Aditya Raj
**Domain:** E-Commerce and Retail Analytics

---


- `Shopper_Spectrum.ipynb` — Main project notebook
- `app.py` — Streamlit web application
- `data/` — Dataset folder
- `models/` — Saved trained models
- `requirements.txt` — Python dependencies

---




pip install -r requirements.txt


streamlit run app.py

---



| Metric | Value |
|---|---|
| Raw Records | 541,909 |
| Clean Records | 392,692 |
| Customers Segmented | 4,338 |
| Silhouette Score | 0.6162 |

---


- ⭐ High-Value — VIP customers, highest spenders
- 🔵 Regular — Steady buyers, loyalty programs
- 🟢 Occasional — Rare buyers, re-engagement needed  
- 🔴 At-Risk — Inactive customers, win-back campaigns

`similarity_matrix.pkl` is not included due to GitHub's 25MB file size limit.
To generate it, run the notebook `Shopper_Spectrum.ipynb` from top to bottom.
It will automatically create all model files inside the `models/` folder.
