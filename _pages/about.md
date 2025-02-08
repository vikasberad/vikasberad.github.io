---
permalink: /
title: "Scalable and Accurate Deep Learning for Electronic Health Records: A Game-Changer in Healthcare"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

![Deep learning in Healthcare](/images/iStock-979576420-768x379.webp)

**Predictive modeling** using Electronic Health Records (EHR) data holds immense potential to revolutionize healthcare by enabling personalized medicine, improving clinical decision-making, and enhancing operational efficiencies. However, the current landscape of predictive analytics faces substantial hurdles. Traditional models require extensive preprocessing, data harmonization, and hand-curated variables, resulting in the loss of valuable clinical information and limiting their scalability.

Rajkomar et al.'s pioneering research, *Scalable and Accurate Deep Learning for Electronic Health Records*, presents a paradigm shift by proposing a novel deep learning framework that harnesses raw EHR data without manual data curation. This approach significantly outperforms traditional predictive models across multiple clinical tasks, including mortality prediction, readmission forecasting, and diagnosis inference.

Through this comprehensive blog, we will explore the key methodologies, results, and implications of this groundbreaking research. We will discuss how the integration of deep learning with EHR systems can enhance healthcare delivery and pave the way for innovative advancements in clinical informatics.

## The Data Challenge in Healthcare

The research, conducted by a team of scientists from Google, University of California, San Francisco, and Stanford University, proposed a radical solution: using deep learning to process entire, raw electronic health records.

### **Key Innovation: Comprehensive Data Representation**
Instead of manually extracting and curating specific variables, the researchers developed a novel approach:

- Represent patients' complete EHR records using the Fast Healthcare Interoperability Resources (FHIR) format.
- Utilize neural networks that can learn complex representations directly from raw data.
- Incorporate comprehensive information, including free-text clinical notes.

## A Deep Learning Breakthrough

The study employed the Fast Healthcare Interoperability Resources (FHIR) format to represent patient records. This approach organizes healthcare data in chronological order without the need for site-specific harmonization.

### **Key Features of FHIR Data Representation:**

- **Temporal Ordering:** Ensures that events in a patient's timeline are accurately recorded.
- **Comprehensive Inclusion:** Incorporates structured data, free-text notes, and other clinical observations.
- **Standardized Format:** Makes it easier for models to learn from diverse data sources.

## Methods

To address the challenges of predictive modeling in healthcare, the researchers adopted a novel approach using deep learning algorithms on raw EHR data. Their methodology aimed to streamline the model-building process, eliminate manual data curation, and leverage the entirety of clinical records, including free-text notes.

### **Data Collection**  
The study utilized de-identified EHR data from two major academic medical centers in the United States — the University of California, San Francisco (UCSF) and the University of Chicago Medicine (UCM). This dataset spanned a combined total of 216,221 hospitalizations and included detailed information such as:  
- Patient demographics  
- Provider orders  
- Diagnoses and procedures  
- Medications  
- Laboratory results  
- Vital signs  
- Free-text clinical notes (UCM data only)  

![EHR Collection](/images/EHRCollection.png)
*Source: Rajkomar et al., 2025, "Scalable and Accurate Deep Learning for Electronic Health Records"*


### **Data Representation**
To represent patients’ records comprehensively, the researchers adopted the Fast Healthcare Interoperability Resources (FHIR) standard. This format structures healthcare data into a chronological sequence of events, allowing the models to process all information from a patient’s record up to any given prediction point.

Key advantages of this representation included:  
- **Temporal sequencing:** Events were organized by time, enabling more accurate temporal predictions.  
- **Comprehensive tokenization:** Each clinical event, including free-text data, was converted into a token for model input.  


![EHR to FHIR](/images/EHRtoFHIR.png)

## Deep Learning Model Variants

To effectively harness the complexity of Electronic Health Records (EHRs) and provide accurate clinical predictions, the researchers explored multiple deep learning model architectures. Each model was designed to address different data challenges and complement the others in capturing meaningful clinical patterns.

### **1. Weighted Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM)**

#### **Overview**  
Recurrent Neural Networks (RNNs) are well-suited for sequential data such as EHRs. However, standard RNNs often fail when processing long sequences due to the vanishing gradient problem. The research team tackled this by using **Long Short-Term Memory (LSTM)** networks, which are designed to remember key information over long time sequences.

#### **Model Design**
- **Sparse Feature Embeddings:**  
  Sparse clinical features (e.g., medications, procedures, lab values) were embedded into `d-dimensional` vectors.  
- **Weighted Feature Averaging:**  
  Features for each time-step (12-hour intervals) were aggregated using learned weights. This process allowed the model to focus on clinically important events while reducing sequence length.  
- **Temporal Processing:**  
  The reduced sequence of embeddings was passed through an **n-layer LSTM**, where each time step's information was selectively retained or discarded using gating mechanisms.

#### **LSTM Gating Mechanisms**
1. **Forget Gate (`ft`)**: Determines which information to discard.  
2. **Input Gate (`it`)**: Decides which new information to store.  
3. **Output Gate (`ot`)**: Controls the information passed to the next step.  

The following equations define LSTM operations:
```markdown
ft = σg(Wfxt + Ufht−1 + bf)  
it = σg(Wixt + Uiht−1 + bi)  
ot = σg(Woxt + Uoht−1 + bo)  
ct = ft · ct−1 + it · σc(Wcxt + Ucht−1 + bc)  
ht = ot · σc(ct)
```
Where:
- `σg`: Sigmoid activation  
- `σc`: Hyperbolic tangent  
- `ct`: Cell state  
- `ht`: Hidden state

![EHR Collection](/images/LSTM.webp)
*Source: Medium.com: An Intuitive Expansion of LSTM(Author: Ottavio Calzone)*

#### **Regularization and Training**
To prevent overfitting and stabilize training:
- **Dropout:** Applied to embeddings and LSTM inputs.  
- **Weight Decay:** L2 regularization penalized large weights.  
- **Gradient Clipping:** Gradients were clipped to maintain stability.  
- **Optimization:** The model was trained using **Adagrad** with hyperparameters tuned via Gaussian processes.

---

### **2. Feedforward Model with Time-Aware Attention (TANN)**

#### **Overview**  
Attention mechanisms allow neural networks to prioritize key data points. The **Time-Aware Attention Neural Network (TANN)** incorporates the dimension of time, focusing on clinically important events close to the prediction point.

#### **Model Design**
1. **Embedding Sequence:**  
   Each clinical event embedding (`Ei`) was associated with a time difference (`∆i`) relative to the prediction point.  
2. **Attention Weights (`βi`)**  
   Attention logits (`αi`) were computed using predefined functions and converted to weights via a softmax function:  
   ```markdown
   βi = e^αi / Σ e^αj
   ```  
3. **Weighted Aggregation:**  
   The embeddings were weighted and summed, along with time-sensitive scalar values, to form the final input to the feedforward network.

#### **Time Functions (`Aj`)**
To capture the importance of time intervals, various functions were applied:
- **Constant (`A(∆) = 1`)**  
- **Linear (`A(∆) = ∆`)**  
- **Logarithmic (`A(∆) = log(∆ + 1)`)**  
- **Piecewise Linear:** With learned slopes  

#### **Training and Hyperparameters**
- Embedding dimensions (`d`) ranged from 16 to 512.  
- Feedforward layers ranged from 1 to 3 with widths up to 512 units.  
- Hyperparameter tuning optimized time functions and network attributes.

---

### **3. Boosted Embedded Time-Series Model**

#### **Overview**  
This model focused on creating interpretable binary decision rules related to clinical events and their timing. By boosting these decision rules, the model refined predictions and identified critical features in patient data.

#### **Model Design**
1. **Binary Decision Rules:**  
   - Example rules include:
     - "Did a lab value exceed a threshold before a specific time?"  
     - "Was a medication administered more than three times?"  
2. **Weighted Rule Aggregation:**  
   Each rule was assigned a weight, and the weighted sum was passed through a softmax layer for prediction.

#### **Boosting Process**
- **Predicate Selection:**  
  In each boosting round, 25,000 random predicates were evaluated for information gain, and the top 50 were selected.  
- **Conjunction Rules:**  
  Secondary predicates were combined to form more complex rules.  
- **Total Boosting Rounds:**  
  100 rounds resulted in 100,000 candidate predicates, reduced through L1 regularization.

  ![EHR Collection](/images/BoostedTime.webp)
  *Source: Your source here*

#### **Final Model Architecture**
The selected predicates were embedded into a 1024-dimensional vector space and fed into a 2-layer feedforward network with **Exponential Linear Unit (ELU)** activations.

---

### **4. Model Ensembling for Optimal Performance**

#### **Why Ensembling?**
Combining multiple models improves robustness, mitigates biases, and reduces prediction errors.

#### **Implementation**
- The predictions from the RNN with LSTM, TANN, and Boosted Time-Series models were averaged to create the final prediction.  
- This ensemble approach leveraged the strengths of each architecture, achieving superior predictive accuracy across clinical tasks.

---

## Model Comparison Table
Below is a comparison of the proposed models against traditional and other advanced models.

| **Model**                      | **Key Strengths**                          | **Performance Metric**      | **Interpretability**        |
|---------------------------------|---------------------------------------------|-----------------------------|-----------------------------|
| RNN with LSTM                  | Captures long-term dependencies           | High AUROC for mortality    | Moderate                    |
| TANN                            | Time-sensitive event prioritization        | Improved readmission AUROC | Moderate                    |
| Boosted Time-Series            | Interpretable binary decision rules        | Effective for lab trends    | High                        |
| Traditional Logistic Regression | Simple, fast                               | Lower AUROC                 | High                        |
| Gradient Boosted Trees          | Handles non-linear relationships well      | Better than logistic        | Moderate                    |
| Deep Ensemble Model            | Combines model strengths                   | Superior across all tasks   | Moderate                    |

This table highlights the strengths of the ensemble approach and the advantages of leveraging multiple neural architectures for EHR-based predictions.



### **Key Takeaways**
- **RNN with LSTM:** Captured long-term dependencies in sequential data.  
- **TANN:** Prioritized time-sensitive events, improving prediction granularity.  
- **Boosted Model:** Provided interpretable decision rules for binary outcomes.  
- **Ensemble:** Combined model strengths for superior clinical predictions.

This multi-model approach exemplified state-of-the-art AI techniques applied to healthcare, significantly improving clinical prediction tasks.


## Results and Performance

The proposed models demonstrated remarkable performance improvements across key clinical prediction tasks. Below are the detailed results:


| **Prediction Task**              | **Performance**                         | **Key Insight**                                                                                       |
|----------------------------------|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| **In-Hospital Mortality Prediction** | AUROC: 0.95 (Hospital A), 0.93 (Hospital B) | The models made accurate predictions 24-48 hours earlier than traditional models, enabling timely interventions. |
| **30-Day Unplanned Readmission Prediction** | AUROC at discharge: 0.77 (Hospital A), 0.76 (Hospital B) | Incorporating free-text notes and comprehensive EHR data improved readmission prediction accuracy over baseline models. |
| **Prolonged Length of Stay (LOS) Prediction** | AUROC: 0.86 (Hospital A), 0.85 (Hospital B) | Early predictions allowed for better hospital resource allocation and discharge planning. |
| **Diagnosis Prediction**          | Weighted AUROC: 0.90 (Both hospitals)    | Accurate multi-diagnosis prediction was complex due to the large number of possible ICD-9 codes.         |

### **1. In-Hospital Mortality Prediction**
- **Performance:** AUROC of **0.95** for Hospital A and **0.93** for Hospital B.  
- **Key Insight:** The models made accurate predictions **24-48 hours earlier** than traditional models, enabling timely interventions.  

### **2. 30-Day Unplanned Readmission Prediction**
- **Performance:** AUROC at discharge of **0.77** for Hospital A and **0.76** for Hospital B.  
- **Key Insight:** Incorporating free-text notes and comprehensive EHR data improved readmission prediction accuracy over baseline models.

### **3. Prolonged Length of Stay (LOS) Prediction**
- **Performance:** Achieved AUROC of **0.86** for Hospital A and **0.85** for Hospital B.  
- **Key Insight:** Early predictions allowed for better hospital resource allocation and discharge planning.

### **4. Diagnosis Prediction**
- **Performance:** Weighted AUROC of **0.90** for both hospitals at discharge.  
- **Challenges:** Accurate multi-diagnosis prediction was complex due to the large number of possible ICD-9 codes.

![EHR Collection](/images/TrainingSet.png)
*Source: Rajkomar et al., 2025, "Scalable and Accurate Deep Learning for Electronic Health Records"*

---

## Key Takeaways from the Results
- **Early Predictions:** The models consistently made predictions earlier than traditional methods, allowing proactive clinical decisions.  
- **High Accuracy:** Significant improvements in AUROC scores across all tasks.  
- **Improved Resource Allocation:** Enhanced predictions of length of stay and readmissions supported better hospital management strategies.

These results highlight the transformative potential of deep learning models for healthcare analytics, setting a new benchmark for predictive modeling in clinical informatics.

![EHR Collection](/images/Accuracy.png)
*Source: Rajkomar et al., 2025, "Scalable and Accurate Deep Learning for Electronic Health Records"*

## Baseline Models

To better understand the performance of the proposed deep learning models, the researchers first developed baseline models using traditional predictive modeling techniques. These baseline models were constructed by selecting commonly used features, based on recent literature, for each prediction task. These hand-engineered features were used exclusively in the baseline models, whereas the deep learning models did not require such extensive feature engineering.

### **1. Mortality Baseline Model – aEWS**

Most traditional models for mortality prediction rely on a small set of clinical features, such as vital signs and lab measurements. For the baseline model, the researchers followed this approach and created a model using the most recent vital signs, including systolic blood pressure, heart rate, respiratory rate, and temperature (converted to Fahrenheit for consistency). Additional lab measurements like white blood cell count, hemoglobin, sodium, creatinine, troponin, lactate, oxygen saturation, glucose, calcium, potassium, chloride, and others were also included.

All values were log-transformed and standardized, with a mean of zero and standard deviation of one based on the development set for each hospital. The model also incorporated the hospital service and patient age.

### **2. Readmission Baseline Model – Modified HOSPITAL Score**

For readmission prediction, the team created a modified version of the HOSPITAL score. This model utilized the most recent values of sodium and hemoglobin (log-transformed and standardized), along with binary indicators for various clinical and demographic factors, such as hospital service, CPT codes during hospitalization, hospitalization length (at least 5 days), prior admissions within the past year, and admission source. The modified HOSPITAL score was designed to predict the likelihood of a 30-day unplanned readmission.

### **3. Length of Stay Baseline Model – Modified Liu**

For predicting prolonged length of stay (LOS), the researchers created a baseline model based on previous work involving general hospital populations. A lasso logistic regression model was constructed using variables such as age, gender, Hierarchical Condition Categories (HCC) codes from the past year, admission source, hospital service, and lab measurements from the mortality baseline model. This model aimed to predict patients likely to experience prolonged hospital stays, enabling better resource allocation.

### **Calibration of Baseline Models**

The calibration curves for the baseline models were evaluated for various prediction tasks, with results from hospitals A and B presented for inpatient mortality, readmission, and length of stay predictions. These calibration curves allowed for an assessment of the models' accuracy and provided insights into how well the traditional models predicted the respective clinical outcomes.

![Calibration Curves](/images/image.png)
*Source: Rajkomar et al., 2025, "Scalable and Accurate Deep Learning for Electronic Health Records"*


---

### **Key Takeaways from Baseline Models**
- **Traditional Feature Engineering:** The baseline models relied heavily on hand-curated features, such as vital signs and lab results, to make predictions.
- **Mortality, Readmission, and LOS Prediction:** Each baseline model targeted a different clinical task, including mortality, readmission, and length of stay, using conventional variables.
- **Performance Comparison:** These baseline models were crucial for understanding how the deep learning models performed against standard approaches that rely on more manual feature selection.

By comparing the performance of these baseline models with the deep learning-based approaches described earlier, the research demonstrated the significant advancements that deep learning can offer, including reducing the need for extensive feature engineering while improving predictive accuracy.


## References

1. “2016 Measure updates and specifications report: hospital-wide all-cause unplanned readmission — version 5.0”. In: Yale–New Haven Health Services Corporation/Center for Outcomes Research & Evaluation (May 2016).
2. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. “Neural Machine Translation by Jointly Learning to Align and Translate”. In: (Sept. 2014). arXiv: 1409.0473 [cs.CL].
3. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. “Neural Machine Translation by Jointly Learning to Align and Translate”. In: (Jan. 2014). arXiv: 1409.0473 [cs.CL].
4. D W Bates et al. “Big data in health care: using analytics to identify and manage high-risk and high-cost patients”. In: Health Aff. 33.7 (2014), pp. 1123–1131.
5. David W Bates et al. “Ten commandments for effective clinical decision support: making the practice of evidence-based medicine a reality”. In: J. Am. Med. Inform. Assoc. 10.6 (Nov. 2003), pp. 523–530.
6. Yoshua Bengio, Aaron Courville, and Pascal Vincent. “Representation Learning: A Review and New Perspectives”. In: (June 2012). arXiv: 1206.5538 [cs.LG].
7. Scott W Biggins et al. “Serum sodium predicts mortality in patients listed for liver transplantation”. In: Hepatology 41.1 (Jan. 2005), pp. 32–39.
8. Ion D Bucaloiu et al. “Increased risk of death and de novo chronic kidney disease following reversible acute kidney injury”. In: Kidney Int. 81.5 (Mar. 2012), pp. 477–485.
9. Federico Cabitza, Raffaele Rasoini, and Gian Franco Gensini. “Unintended Consequences of Machine Learning in Medicine”. In: JAMA (July 2017).
10. Vineet Chopra and Laurence F McMahon Jr. “Redesigning hospital alarms for patient safety: alarmed and potentially dangerous”. In: JAMA 311.12 (Mar. 2014), pp. 1199–1200.
11. CMS’ ICD-9-CM to and from ICD-10-CM and ICD-10-PCS Crosswalk or General Equivalence Mappings. http://www.nber.org/data/icd9-icd-10-cm-and-pcs-crosswalk-generalequivalence-mapping.html. Accessed: 2017-7-21.
12. Jacques Donzé et al. “Potentially avoidable 30-day hospital readmissions in medical patients: derivation and validation of a prediction model”. In: JAMA Intern. Med. 173.8 (Apr. 2013), pp. 632–638.
13. Jacques Donzé et al. “Potentially avoidable 30-day hospital readmissions in medical patients: derivation and validation of a prediction model”. In: JAMA Intern. Med. 173.8 (22 4 2013), pp. 632–638.
14. Barbara J Drew et al. “Insights into the problem of alarm fatigue with physiologic monitor devices: a comprehensive observational study of consecutive intensive care unit patients”. In: PLoS One 9.10 (Oct. 2014), e110274.
15. Gabriel J Escobar et al. “Nonelective Rehospitalizations and Postdischarge Mortality: Predictive Models Suitable for Use in Real Time”. In: Med. Care 53.11 (Nov. 2015), pp. 916–923.
16. Andrea Frome et al. “DeViSE: A Deep Visual-Semantic Embedding Model”. In: Advances in Neural Information Processing Systems 26. Ed. by C J C Burges et al. Curran Associates, Inc., 2013, pp. 2121–2129.
17. Benjamin A Goldstein et al. “Opportunities and challenges in developing risk prediction models with electronic health records data: a systematic review”. In: J. Am. Med. Inform. Assoc. 24.1 (Jan. 2017), pp. 198–208.
18. Daniel Golovin et al. “Google Vizier: A Service for Black-Box Optimization”. In: Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2017.
19. Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press, 2016.
20. Kevin Grumbach, Catherine R Lucey, and S Claiborne Johnston. “Transforming From Centers of Learning to Learning Health Systems: The Challenge for Academic Health Centers”. In: JAMA 311.11 (Mar. 2014), pp. 1109–1110.
21. Varun Gulshan et al. “Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs”. In: JAMA 316.22 (Dec. 2016), pp. 2402–2410.
22. John D Halamka and Micky Tripathi. “The HITECH Era in Retrospect”. In: N. Engl. J. Med. 377.10 (Sept. 2017), pp. 907–909.
23. Health Level 7. http://hl7.org/fhir/. Accessed: 2017-8-3. Apr. 2017.
24. Sepp Hochreiter and Jürgen Schmidhuber. “Long Short-Term Memory”. In: Neural Comput. 9.8 (Nov. 1997), pp. 1735–1780.
25. P James et al. “Relation between troponin T concentration and mortality in patients presenting with an acute stroke: observational study”. In: BMJ 320.7248 (Mar. 2000), pp. 1502–1504.
26. J Larry Jameson and Dan L Longo. “Precision medicine–personalized, problematic, and promising”. In: N. Engl. J. Med. 372.23 (June 2015), pp. 2229–2234.
27. Paul R Kalra et al. “Hemoglobin and Change in Hemoglobin Status Predict Mortality, Cardiovascular Events, and Bleeding in Stable Coronary Artery Disease”. In: Am. J. Med. (19 1 2017).
28. Devan Kansagara et al. “Risk prediction models for hospital readmission: a systematic review”. In: JAMA 306.15 (Oct. 2011), pp. 1688–1698.
29. Kirsi-Maija Kaukonen et al. “Systemic inflammatory response syndrome criteria in defining severe sepsis”. In: N. Engl. J. Med. 372.17 (Apr. 2015), pp. 1629–1638.
30. John Kellett and Arnold Kim. “Validation of an abbreviated VitalpacTM Early Warning Score (ViEWS) in 75,419 consecutive admissions to a Canadian regional hospital”. In: Resuscitation 83.3 (Mar. 2012), pp. 297–302.
31. Lauren J Kim et al. “Cardiac troponin I predicts short-term mortality in vascular surgery patients”. In: Circulation 106.18 (29 10 2002), pp. 2366–2371.
32. Andrew A Kramer and Jack E Zimmerman. “Assessing the calibration of mortality benchmarks in critical care: The Hosmer-Lemeshow test revisited”. In: Crit. Care Med. 35.9 (Sept. 2007), pp. 2052–2056.
33. Harlan M Krumholz. “Big data and new knowledge in medicine: the thinking, training, and tools needed for a learning health system”. In: Health Aff. 33.7 (July 2014), pp. 1163–1170.
34. Harlan M Krumholz, Sharon F Terry, and Joanne Waldstreicher. “Data Acquisition, Curation, and Use for a Continuously Learning Health System”. In: JAMA 316.16 (Oct. 2016), pp. 1669–1670.
35. Jean-Philippe Lafrance and Donald R Miller. “Acute kidney injury associates with increased long-term mortality”. In: J. Am. Soc. Nephrol. 21.2 (Feb. 2010), pp. 345–352.
36. Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. “Deep learning”. In: Nature 521.7553 (May 2015), pp. 436–444.
37. Vincent Liu et al. “Length of stay predictions: improvements through the use of automated laboratory and comorbidity variables”. In: Med. Care 48.8 (Aug. 2010), pp. 739–744.
38. Steve Lohr. “For Big-Data Scientists, ‘Janitor Work’ Is Key Hurdle to Insights”. In: The New York Times (Aug. 2014).
39. Mingshan Lu et al. “Systematic review of risk adjustment models of hospital length of stay (LOS)”. In: Med. Care 53.4 (Apr. 2015), pp. 355–365.
40. Joshua C Mandel et al. “SMART on FHIR: a standards-based, interoperable apps platform for electronic health records”. In: J. Am. Med. Inform. Assoc. 23.5 (Sept. 2016), pp. 899–908.
41. Ziad Obermeyer and Ezekiel J Emanuel. “Predicting the Future — Big Data, Machine Learning, and Clinical Medicine”. In: N. Engl. J. Med. 375.13 (2016), pp. 1216–1219.
42. Ravi B Parikh, J Sanford Schwartz, and Amol S Navathe. “Beyond Genes and Molecules - A Precision Delivery Initiative for Precision Medicine”. In: N. Engl. J. Med. 376.17 (Apr. 2017), pp. 1609–1612.
43. Fabian Pedregosa et al. “Scikit-learn: Machine Learning in Python”. In: J. Mach. Learn. Res. 12.Oct (2011), pp. 2825–2830.
44. Adler Perotte et al. “Diagnosis code assignment: models and evaluation metrics”. In: J. Am. Med. Inform. Assoc. 21.2 (Mar. 2014), pp. 231–237.
45. Gil Press. Cleaning Big Data: Most Time-Consuming, Least Enjoyable Data Science Task, Survey Says. https://www.for