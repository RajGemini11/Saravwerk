# Coupon Acceptance Analysis

This repository contains an exploratory analysis of the UCI "Coupons" dataset (assignment 5.1). The primary notebook, `prompt_coupon_selection.ipynb`, walks through data loading, basic cleaning, visualization, and targeted investigations into which drivers accept coupons (with a focus on Bar and Coffee House coupons).

## Dataset
  - `C:/Users/Raj03/Downloads/assignment5_1_starter/data/Coupons.csv`

## What I did (high level)

- Loaded the CSV into a pandas DataFrame.
- Filled missing values for visit-frequency columns (e.g., `Bar`, `CoffeeHouse`, `CarryAway`, `RestaurantLessThan20`, `Restaurant20To50`) with the string `never`.
- Produced basic visualizations: a bar plot of coupon types and a histogram of the `temperature` column.
- Performed focused analyses:
  - Bar coupons: acceptance increases for users with higher bar visit frequency, when there are no minor passengers, and for certain age groups.
  - Coffee House coupons: acceptance aligns with habitual coffee consumption, routine-based trips (e.g., morning), and when the trip context has no urgent destination.

## Key findings / hypotheses

- Bar coupons are more likely to be accepted by drivers who already frequent bars (1+ times/month), are not accompanied by kids, and fall into certain age brackets — the coupon often acts as a small nudge for an already likely behavior.
- Coffee House coupon acceptance is driven more by existing coffee habits and routine alignment (time of day, destination). For habitual coffee consumers, coupons reinforce routine stops rather than triggering large detours.

These are exploratory observations — not formal statistical tests. Use the notebook to refine and validate them further.

Jupyter Notebook Link from Git Public Repo:
https://github.com/RajGemini11/Saravwerk/blob/main/prompt_coupon_selection.ipynb