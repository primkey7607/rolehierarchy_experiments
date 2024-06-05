# Role Hierarchy Experiments
This repository contains all the code needed to generate a benchmark for testing synthesis and auditing of access control privilege and role hierarchy implementation on a Postgres database.

One should be able to reproduce our results by running each of our scripts in a one-off fashion. We will explain each script below:

1. amazon_access.py: organizes the Amazon Access dataset into a tree.
2. rolehier_gen.py: replaces IDs with gpt-4o-generated role labels and descriptions.
