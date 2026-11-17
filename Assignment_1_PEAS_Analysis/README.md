# Assignment 1: AI/ML Applications and PEAS Analysis

**Course:** Artificial Intelligence & Machine Learning  
**Author:** Shruti Siddheshwar Thorat  
**PRN:** 202401040053  
**Roll No:** 16  

---

## 📌 Overview

This study explores the foundational concepts of Artificial Intelligence (AI), Machine Learning (ML), and Rational Intelligent Agents through the **PEAS (Performance Measure, Environment, Actuators, Sensors)** framework across two distinct real-world domains:
1. **AI-Based Medical Diagnosis System** (Healthcare)
2. **AI-Based Smart Museum & Heritage Preservation System** (Cultural Preservation)

---

## 🧠 System 1: AI-Based Medical Diagnosis System

### Objectives
- Improve diagnostic support and reduce diagnostic errors.
- Detect diseases at earlier stages and identify high-risk patients.
- Accelerate data analysis across complex medical imagery and lab tests.
- Support clinical workflows and remote telemedicine without replacing certified healthcare practitioners.

### 5 Intelligent Agents
1. **Patient Monitoring Agent:** Continuously tracks vital signs and physiological states.
2. **Medical Image Analysis Agent:** Processes X-rays, CT scans, MRIs, and ultrasounds to detect anomalies.
3. **Disease Detection Agent:** Synthesizes symptoms, clinical history, and laboratory readings to identify diseases.
4. **Risk Prediction Agent:** Forecasts likelihood of complications, deterioration, or readmission.
5. **Treatment Recommendation Agent:** Formulates evidence-based clinical protocols and treatment options for doctor review.

### 5 Environments
1. **Hospital Ward:** In-patient monitoring under continuous clinical observation.
2. **Emergency Department:** Fast-paced, triage-heavy setting with incoming critical patients.
3. **Diagnostic Laboratory:** Processing blood, urine, microbiology, and biochemical assays.
4. **Medical Imaging Center:** High-resolution radiological scans requiring expert inspection.
5. **Telemedicine / Remote Healthcare:** Home-based monitoring via digital consultation and consumer wearables.

### PEAS Summary Table
| No. | Agent | Environment | Performance Measure (P) | Environment (E) | Actuators (A) | Sensors (S) |
|---|---|---|---|---|---|---|
| 1 | Patient Monitoring Agent | Hospital Ward | Monitoring accuracy, early warning rate, low false alarms | In-patient ward, beds, nursing staff | Alerts, dashboard update, staff notification | ECG, BP monitor, SpO2, temperature sensors |
| 2 | Patient Monitoring Agent | Emergency Department | Speed of detection, triage accuracy, low false negatives | ER triage, trauma units, emergency physicians | Urgent alerts, prioritized queues | ECG, vitals monitor, triage entry system |
| 3 | Medical Image Analysis Agent | Medical Imaging Center | Image analysis precision, abnormality detection rate | Radiology department, PACS imaging database | Visual heatmaps, anomaly bounding boxes, draft reports | High-res X-ray, CT scanner, MRI, ultrasound data |
| 4 | Medical Image Analysis Agent | Emergency Department | Processing latency, critical trauma detection | Emergency radiology, acute trauma cases | Stat notifications to ER surgeons, triage flags | Portable X-ray, CT scanner feeds |
| 5 | Disease Detection Agent | Diagnostic Laboratory | Differential diagnosis accuracy, assay classification | Pathology lab, specimen analyzers | Flag abnormal labs, diagnostic suggestions | Blood analyzer, urinalysis, biochemical assays |
| 6 | Disease Detection Agent | Telemedicine / Remote | Diagnostic recommendation safety, completeness | Patient home, remote tele-consult platform | Health risk report, doctor appointment referral | Patient self-reported symptoms, digital logs |
| 7 | Risk Prediction Agent | Hospital Ward | Complication prediction accuracy, advance warning time | In-patient ward, electronic medical records (EMR) | Proactive ICU transfer alert, monitoring increase | EMR trend data, real-time vitals, lab results |
| 8 | Risk Prediction Agent | Emergency Department | Triage risk scoring accuracy, missed emergency reduction | Emergency intake, overcrowded waiting rooms | Color-coded urgency levels, resuscitation alerts | Incoming vitals, triage questionnaires |
| 9 | Treatment Recommendation Agent | Diagnostic Laboratory | Guideline adherence, medication safety, test relevance | Lab information system (LIS), pathologist team | Suggested confirmatory panels, antibiotic sensitivity | Culture reports, biochemical profiles |
| 10 | Treatment Recommendation Agent | Telemedicine / Remote | Consultation relevance, referral safety, clarity | Telehealth mobile app, primary care physician | Structured symptom summary, self-care guidelines | Wearable vitals, consultation logs |

---

## 🏛️ System 2: AI-Based Smart Museum & Heritage Preservation System

### Objectives
- Automate non-destructive monitoring of ancient paintings, sculptures, and heritage architecture.
- Track micro-climate variables (temperature, humidity, UV light, volatile compounds) to avoid decay.
- Optimize visitor flow and prevent overcrowding around delicate artifacts.
- Strengthen site security, preventing unauthorized access, theft, and vandalism.

### 5 Intelligent Agents
1. **Artifact Condition Monitoring Agent:** Regularly inspects degradation, flaking, and discoloration.
2. **Artifact Damage Detection Agent:** Detects hairline cracks, structural shifts, and physical impacts.
3. **Climate Control Agent:** Dynamically orchestrates HVAC, humidifiers, and air purification.
4. **Visitor Flow Management Agent:** Monitors footfall distribution and manages queueing.
5. **Security & Intrusion Detection Agent:** Identifies security breaches, perimeter crossings, and suspicious behaviors.

### 5 Environments
1. **Art Gallery:** Public exhibition space showcasing delicate canvas paintings and sculptures.
2. **Historical Artifact Storage Room:** Highly restricted archival vaults with strict preservation criteria.
3. **Museum Exhibition Hall:** High-traffic open spaces with multiple simultaneous exhibits.
4. **Archaeological Heritage Site:** Outdoor historical ruins vulnerable to weather and erosion.
5. **Museum Security & Control Room:** Central supervisory hub integrating all video feeds and sensor telemetry.

### PEAS Summary Table
| No. | Agent | Environment | Performance Measure (P) | Environment (E) | Actuators (A) | Sensors (S) |
|---|---|---|---|---|---|---|
| 1 | Condition Monitoring Agent | Art Gallery | Assessment accuracy, coverage, false alert rate | Indoor gallery, changing visitor density | Warning alerts, conservator work-order | High-res RGB cameras, lux/UV sensors |
| 2 | Condition Monitoring Agent | Archaeological Site | Coverage percentage, weathering detection | Outdoor monuments, weather extremes | GIS anomaly map, drone flight schedule | Drone multispectral cameras, LiDAR, GPS |
| 3 | Damage Detection Agent | Storage Room | Crack detection sensitivity, inspection speed | Archival vault, storage racks, sealed cases | Damage flag in inventory, inspection request | 3D laser scanners, macro cameras, RFID |
| 4 | Damage Detection Agent | Art Gallery | Impact detection latency, visual evidence quality | Display cases, public exhibition galleries | Guard dispatch alert, instant camera zoom | CCTV feeds, optical vibration sensors |
| 5 | Climate Control Agent | Storage Room | Humidity/temperature stability, energy efficiency | Enclosed vault, sensitive organic artifacts | HVAC micro-adjustments, dehumidifier trigger | Ambient temperature, relative humidity, VOC sensors |
| 6 | Climate Control Agent | Art Gallery | Environmental stability amidst fluctuating crowds | Public gallery rooms, external doors | Smart airflow vents, climate alert dispatch | Air quality sensors, temperature, CO2 probes |
| 7 | Visitor Flow Agent | Exhibition Hall | Counting accuracy, queue minimization, safety | Multi-gallery exhibition halls, doorways | Digital rerouting signage, barrier gates | Overhead people counters, thermal sensors |
| 8 | Visitor Flow Agent | Archaeological Site | Congestion avoidance, perimeter enforcement | Unpaved pathways, open-air ruins | Mobile app route recommendation, alerts | Drone cameras, GPS tracking, path infrared sensors |
| 9 | Security & Intrusion Agent | Control Room | Threat detection speed, minimal false alarms | Central command station, video walls | Alarm sirens, automatic door deadbolts | Motion detectors, CCTV video analytics |
| 10 | Security & Intrusion Agent | Storage Room | Unauthorized breach detection, response latency | High-security vault access vestibule | Door interlocks, security team page, video capture | Biometric readers, RFID tags, PIR sensors |

---

## ⚖️ Cross-Domain Comparison

| Parameter | AI-Based Medical Diagnosis System | AI-Based Smart Museum & Heritage Preservation |
|---|---|---|
| **Primary Objective** | Support clinical decisions, save lives, minimize morbidity | Preserve cultural heritage, prevent physical decay, manage visitors |
| **Core AI Disciplines** | Deep learning, computer vision, Bayesian risk modeling | Computer vision, anomaly detection, IoT sensor fusion, drone analytics |
| **Sensor Types** | Biosensors, physiological monitors, radiological scanners | Photogrammetry, 3D laser scanners, climate IoT probes, surveillance CCTV |
| **Actuator Outputs** | Clinical diagnostic alerts, dosage checks, doctor summaries | HVAC modulation, perimeter alarms, crowd guidance signs |
| **Worst-case Failure** | Missed critical diagnosis or incorrect medication recommendation | Irreversible destruction of priceless historical artifacts |
| **Human Role** | Ultimate decision-maker (licensed doctor/physician) | Conservators, curators, and security personnel |
