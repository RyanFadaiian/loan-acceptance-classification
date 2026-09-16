# Loan acceptance experiment

I compared a depth-6 decision tree and a 50-tree forest, each with and without SMOTE. The target was personal-loan offer acceptance. The 5,000-row data contained 480 acceptors, so always predicting nonacceptance would already achieve 90.4% accuracy while missing every acceptor.

The split was stratified: 3,750 training rows and 1,250 test rows. Scaling used training statistics only. SMOTE used five neighbors and balanced the training classes to 3,390 each. All four configurations used the original test distribution. Forest predictions averaged tree probabilities rather than taking a majority of hard class votes.

The [recorded results](results/recorded_metrics.csv) show the main tradeoff. Forest recall increased from 86.7% to 96.7% with SMOTE, while precision fell from 98.1% to 81.1%. Both models had lower F1 after oversampling. A high-recall model may suit inexpensive outreach, while high precision may matter more with limited capacity; no costs or profitability were estimated here.

ROC AUC and F1 need not move together: AUC evaluates score ranking across thresholds, while F1 evaluates a particular classification rule. High accuracy alone does not diagnose overfitting. These metrics also do not establish that tree probability estimates are calibrated.

Limitations include a single inspected split, rounded historical metrics, and synthetic fractional categorical features. The course tree and SMOTE implementations were supplied. My later work completed the forest and SMOTE integration and interpreted the results with assistant guidance.

During packaging, the saved notes claimed local seeded bootstrapping, but the script used NumPy's global random generator. The upload copy corrects this. Historical values are preserved rather than silently replaced; fresh outputs must be reported as a separate run. No conclusion about statistical significance is made.
