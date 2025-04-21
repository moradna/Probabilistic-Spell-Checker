# Probabilistic-Spell-Checker

This project implements a basic **spell checker** based on the **noisy channel model** and an **n-gram language model**.

It supports both word-level and character-level models, and uses **confusion matrices** to simulate realistic spelling errors.

## 💡 How it works

1. Builds an n-gram language model from input text.
2. Detects misspelled words (OOV).
3. Suggests corrections based on:
   - Edit distance (1 or 2)
   - Language model probability
   - Error likelihood from confusion matrices
  
##  📁 Project Structure

project-root/
├── spell_checker.py        # Main implementation
├── spelling_confusion_matrices.py  # Error tables (external)
├── corpus/                 # Text data for training LM
├── test_cases/             # Sample inputs and evaluations
└── README.md
