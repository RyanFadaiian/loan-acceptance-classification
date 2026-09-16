# Included data

loan.csv is the course-supplied 5,000-row Universal Bank personal-loan dataset. I confirmed permission to publish the class materials. It is included, so no dataset download is needed. Its exact public upstream origin has not been independently verified.

Required columns: ID, Age, Experience, Income, ZIP Code, Family, CCAvg, Education, Mortgage, Personal Loan, Securities Account, CD Account, Online, CreditCard.

Personal Loan is the target: 1 means the customer accepted the offer. The script drops ID and ZIP Code, takes absolute values of Experience, creates a stratified 75/25 split with seed 42, and fits scaling statistics on training rows only. SMOTE is applied only to training data.

The file hash is recorded in results/data_manifest.json. No pretrained model is required. Run python experiment.py from this folder after installing requirements.txt.
