# Employee Onboarding Chatbot

A conversational AI solution designed to streamline the onboarding process for new employees. This chatbot assists new hires by answering common HR questions, guiding them through company policies, and providing support during their early days at work.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Initialize](#project-initialize)
- [Installation](#installation)
- [Data Sample](#the-data-format)

---

## Overview

Employee Onboarding Chatbot is built to:
- **Welcome and Guide New Hires:** Provide personalized greetings, share company culture, and clarify role expectations.
- **Answer HR FAQs:** Respond to queries about policies, benefits, document submissions, and other onboarding-related topics.
- **Reduce Administrative Workload:** Automate routine HR tasks, allowing HR teams to focus on strategic activities.
- **Continuously Improve:** Learn from user interactions to enhance response accuracy over time.

---

## Features

- **Conversational Interface:** Natural language understanding for human-like interactions.
- **HR-Specific Intents:** Supports recruitment, policy queries, leave requests, and more.
- **Integration Ready:** Easily integrates with your HR systems and internal data sources.
- **Continuous Learning:** Retrain and fine-tune the model with new onboarding data.
- **Multi-Channel Deployment:** Deployable on web, Slack, or any messaging platform.

---

## Technologies Used

- **Programming Language:** Python
- **NLP & Chatbot Framework:** NLU model (DistilBERT)
- **Machine Learning Libraries:** Pytorch, scikit-learn
- **Data Handling:** Pandas, NumPy
- **Deployment:** Flask for API integration

---
## Project Initialize

1. **Directory Creation**
   
   ```bash
   mkdir chatbot_project
   cd chatbot_project
2. **Environment Creation**

   ```bash
   conda create -n chatbot_env python=3.10
3. **Activate Environment**
   
   ```bash
   conda activate chatbot_env
4. **Install Dependencies and Package**
   
   ```bash
   pip install ...

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/JinMaru01/Employee_onboarding_chatbot.git 
   cd Employee-onboarding-chatbot

   python3 -m venv venv
2. **Activate the virtual environment**
    ```bash
    source venv/bin/activate  # On Windows: venv\Scripts\activate

This template is fully formatted in Markdown and ready to be saved as your repository's `README.md`. Adjust any section as needed to match your project's specifics.
## The Data Format

1. **Intent Dataset**

   ```bash
   {
   "intent": "ask_about_nine_principles",
   "question": "What is Principle 6?"
   }

2. **Entity Dataset**
   
   ```bash
    {
       "text": "If I want to resign what I need to know?",
       "intent": "ask_contract_termination",
       "entities": [
         {
           "start": 13,
           "end": 19,
           "label": "TERMINATION_TYPE",
           "value": "resign"
         }
       ]
     }

3. **Combine both tasks**

   ```bash
   {
     "text": "Is maternity insurance covered in our health plan?",
     "intent": "benefit_procedures",
     "tokens": [
       "be",
       "maternity",
       "insurance",
       "cover",
       "in",
       "our",
       "health",
       "plan"
     ],
     "tags": [
       "O",
       "B-LEAVE_TYPE",
       "B-BENEFIT_TYPE",
       "O",
       "O",
       "O",
       "B-BENEFIT_TYPE",
       "O"
     ]
   }
