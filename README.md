![picture](./assets/R.jpg)

# Aligning Language Models to Individual-Practitioner-Level TCM Prescription Preferences Using Synthetic Clinical Cases


## Generation Task:
1. Generation Task 1: Synthetic TCM clinical cases and their verifiable questions
2. Generation Task 2: Verifiable general-purpose medical questions
3. Generation Task 3: CoT responses for verifiable questions on prescription recommendation (individual-preference prescriptions or general-purpose prescription) 
4. Generation Task 4: CoT responses for verifiable questions on general medical knowledge


## SFT
The training data is accessible at https://huggingface.co/datasets/asterhouse/SyntheticTCMCases.


## Eval
1. benchmark: TCM-Ladder
2. case_eval: 104 real-world clinical cases
