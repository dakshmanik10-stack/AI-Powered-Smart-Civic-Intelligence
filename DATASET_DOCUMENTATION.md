# DATASET DOCUMENTATION & AUDIT
**Project Name:** AI-Powered Civic Complaint Intelligence System  
**Dataset Filename:** `dataset/municipal_training_set_1100.csv`  
**Date:** October 29, 2026  
**Status:** Verified & Formally Documented  

---

## 1. Dataset Origin & Authenticity

| Attribute | Verified Fact |
| :--- | :--- |
| **Exact Filename** | `municipal_training_set_1100.csv` |
| **Dataset Title** | CivicAI Municipal Training Set (1100 Records) |
| **Original Publisher / Author** | Shubham Suneja (*Reinvent the Wheel* / Tech Engineering Publication) |
| **Source Series** | *"Building CivicAI Part 1: When No Dataset Exists, You Build One"* (`ritw.dev`) |
| **Nature of Data** | **Curated Synthetic Benchmark Dataset** (Synthesized using structured civic issue templates based on municipal 311 patterns) |
| **License** | Open Educational / Academic Demonstration (No commercial restriction declared; no formal license badge attached) |
| **Verified vs Unverified** | **Verified:** Filename, schema, column structure, 1100 rows, blog source origin. **Unverified:** No live official government API endpoint for this specific CSV. |

### Does this represent real Indian municipal complaints?
- **Honest Assessment:** **No, geographically it does not.** The geographic attributes in this dataset refer entirely to US cities (`New York`: 212, `Chicago`: 211, `Houston`: 203, `Phoenix`: 191, `Los Angeles`: 149) with US zip codes and streets.
- **Problem Domain Alignment:** However, the *linguistic descriptions of civic problems* (potholes, open drains, garbage overflow, flickering streetlights, water pipeline bursts, hazardous hanging wires) are universal urban infrastructure issues that directly mirror municipal grievances faced across Indian cities.
- **Viva Defense Point:** When asked by examiners: *"Aapka dataset Indian hai ya foreign?"* -> You must answer truthfully:
  > *"Sir, public Indian municipal grievance datasets (like CPGRAMS or municipal portals) are largely restricted due to citizen privacy and PII laws. Therefore, for training and testing our AI pipeline, we utilized a curated 1,100-complaint benchmark dataset based on the standard 311 civic schema from the CivicAI technical benchmark series. The complaints represent universal civic challenges (water, roads, sanitation, electrical, traffic), but for live production, the pipeline is designed to intake Indian addresses and pincodes (as shown in our Streamlit portal)."*

---

## 2. Dataset Dimensions & Schema

- **Total Rows:** 1,100 rows
- **Total Columns:** 26 columns
- **Memory Footprint:** ~487 KB

### Complete Column Dictionary:

| # | Column Name | Data Type | Null Count | Description | Role in ML Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `report_id` | String | 0 | Unique complaint ID (e.g., `RPT-354769`) | Identifier |
| 2 | `resident_id` | String | 0 | Citizen ID of submitter | Metadata |
| 3 | `resident_name` | String | 63 | Name of resident (demo/synthetic) | Metadata |
| 4 | `resident_email` | String | 63 | Email address of resident | Metadata |
| 5 | `resident_phone` | String | 78 | Phone number of resident | Metadata |
| 6 | `issue_type` | String | 0 | Category tag (`garbage`, `pothole`, etc.) | Secondary Label |
| 7 | **`issue_description`** | String | 0 | Detailed text complaint | **PRIMARY ML FEATURE ($X$)** |
| 8 | `issue_status` | String | 0 | Status (`in_progress`, `closed`, `new`) | Status Tracking |
| 9 | `report_datetime` | DateTime | 0 | Timestamp of report filing | Temporal |
| 10 | `location_street_address` | String | 0 | Address string | Geographic |
| 11 | `location_city` | String | 0 | City name (NYC, Chicago, etc.) | Geographic |
| 12 | `location_state` | String | 0 | State abbreviation | Geographic |
| 13 | `location_postal_code` | String | 0 | Postal zip code | Geographic |
| 14 | `location_country` | String | 0 | Country (`USA`) | Geographic |
| 15 | `latitude` | Float | 0 | GPS Latitude coordinate | Spatial GIS |
| 16 | `longitude` | Float | 0 | GPS Longitude coordinate | Spatial GIS |
| 17 | `photo_url` | String | 454 | Web link to photo attachment | Media attachment |
| 18 | **`municipal_department`** | String | 7 | Target department for grievance | **PRIMARY TARGET LABEL ($y$)** |
| 19 | `resolution_datetime` | DateTime | 830 | Date/time issue was solved | Resolution metric |
| 20 | `resolution_notes` | String | 830 | Department resolution comments | Resolution note |
| 21 | `is_anonymous` | Boolean | 0 | Anonymous filing flag | Privacy flag |
| 22 | `priority_level` | String | 0 | Ordinal priority (`low`, `medium`, `high`, `critical`) | **Priority Target ($y_{priority}$)** |
| 23 | `priority_score` | Float | 0 | Calculated score (1.0 to 10.0) | Numerical Priority |
| 24 | `severity_score` | Integer | 0 | Hazard severity (1 to 10) | Priority Feature |
| 25 | `area_importance` | Integer | 0 | Criticality of area (1 to 10) | Priority Feature |
| 26 | `citizen_reports_count` | Integer | 0 | Upvotes/complaints count (1 to 50) | Priority Feature |

---

## 3. Data Quality, Anomalies & Leakage Discovery

### 3.1 Critical Discovery: The 653 Duplicate Descriptions
- Out of 1,100 records, there are **only 447 unique text descriptions**.
- 653 records are repeated complaint texts generated across different resident IDs and locations to simulate recurring citizen grievances.
- **Data Leakage Danger:** If a simple `train_test_split(test_size=0.2)` is applied on raw rows without deduplicating or grouping, identical complaint sentences appear in both the training set and the test set. The model ends up testing on sentences it has already seen, causing **test contamination and artificial accuracy inflation**.
- **Engineering Solution:** 
  1. For authentic model evaluation, we train on unique complaint descriptions or enforce a **GroupKFold/GroupSplit** by description hash so no sentence appears in both splits.
  2. In the application, identical complaints are leveraged for **Duplicate Complaint Detection (Phase 6)**.

### 3.2 Label Fragmentation in Target (`municipal_department`)
The dataset originally contains 25 distinct departmental labels, many of which are redundant micro-labels for the same real-world department:

```text
- Water: 'Water Department' (153), 'Water Management' (5), 'Water Services' (5), 'Waterworks' (5)
- Roads: 'Roads & Highways' (129), 'Road Maintenance' (9), 'Roads' (6), 'Public Works' (20)
- Electricity: 'Electrical Department' (115), 'Electrical Services' (20), 'Electrical' (6)
- Sanitation: 'Sanitation' (177), 'Waste Management' (5)
- Parks: 'Parks & Recreation' (7), 'Parks Department' (2)
- Safety/Law: 'Public Safety' (120), 'Noise Control' (11), 'Code Enforcement' (3)
```

### 3.3 Authoritative Standardized Department Mapping
To build a robust, production-grade model with strong generalization and high macro-F1 score, we define a clear, documented 8-department taxonomy:

| Standardized Department | Original Classes Merged | Total Clean Samples |
| :--- | :--- | :--- |
| **Sanitation & Waste Management** | `Sanitation`, `Waste Management` | 182 |
| **Water Supply & Management** | `Water Department`, `Water Management`, `Water Services`, `Waterworks` | 168 |
| **Roads & Infrastructure** | `Roads & Highways`, `Road Maintenance`, `Roads`, `Public Works` | 164 |
| **Traffic Department** | `Traffic Department` | 135 |
| **Drainage Authority** | `Drainage Authority` | 135 |
| **Electrical Department** | `Electrical Department`, `Electrical Services`, `Electrical` | 141 |
| **Public Safety & Enforcement** | `Public Safety`, `Noise Control`, `Code Enforcement` | 134 |
| **Parks & Community Services** | `Parks & Recreation`, `Parks Department`, `Community Services`, `Community Affairs`, `Event Management` | 33 |

*(Note: 7 rows with NaN departments and 1 row with 'Postal Services' are discarded).*

---

## 4. Input Features & Target Labels

### For Department Classification (NLP Pipeline):
- **Input Feature ($X$):** Cleaned `issue_description`
- **Target Label ($y$):** Standardized `municipal_department` (8 classes)
- **Evaluation Strategy:** Stratified Train/Test Split (80% Train, 20% Test) with Random State = 42 and zero train/test leakage.

### For Priority Prediction:
- **Input Features ($X$):** `severity_score`, `area_importance`, `citizen_reports_count`, plus text urgency signals.
- **Target Label ($y$):** `priority_level` (`low`, `medium`, `high`, `critical`).
