"""Supporting code for Lab 2: referral routes into English Wikipedia.

The package keeps the heavy Spark work and the figures out of the report
notebook (``main.ipynb``), which imports these modules and only shows
small tables, figures and the text that interprets them.

Modules
-------
config
    Every path, threshold and Spark setting in one place.
session
    Starts Spark with limits that fit the Jojie server.
preparation
    Finds the files, stages and profiles each month, and builds the
    month x article x route table.
routes
    The single CASE expression for routes and the share queries.
cohorts
    Popularity rankings and cohort tables.
decomposition
    Splits the change in search share into its sources.
explore
    Descriptive tables for the raw-data section.
checks
    Release-artifact flags, robustness reruns and reconciliation checks.
timing
    Wall-clock timing of the slow steps.
plots
    Every figure in the report.
report
    Shows tables in the notebook with readable numbers.
"""

__version__ = '1.0.0'
