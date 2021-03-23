# Debugging ML systems - A simple guide.

# Goal: Build a model, e.g a language Model

# Metrics to look at:
# 1. Human-level performance or some benchmark
# 2. Trainig set error
# 3. Dev set error

# --------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# --------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Work flow for debugging ML systems:

# training_error = final_error_on_training_set
# high_training_error = value_perceived_as_high_on_training_set 

# if training_error > high_training_error:
# 	# You have a high bias problem.
# 	# Do one or multiple of the steps below until training error 
# 	# is low.
# 	1. Try a bigger model
# 	2. Train for longer
# 	3. New model architecture


# dev_error = final_error_on_dev/evaluation_set
# high_dev_error = value_perceived_as_high_on_dev_set

# if dev_error > high_dev_error:
# 	# You have a high variance problem.
# 	# Do one or multiple of the steps below until dev error is low
# 	1. Get more data
# 	2. Add regularization
# 	3. New model architecture


# ------------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# __DATA__

# -> Make sure Dev and test set are from same distribution.
	# This is because an ML team will spend a lot of time optimizing/tunning for the Dev set.
	# If the test set has a different distribution from the Dev set then the model simply will not work.


# ------------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# __Human level perfomance__

# -> Human level perfomance as a benchmark

# 	|                               
# 	|       ________________________  model perfomance                          
# 	|      /                             
# 	|...../.......................    human-level perfomance                         
# 	|    /                            
# 	|   /                           
# 	|  /                             
# 	| /                               
# 	|/______________________________
# 				TIME

# -While worse than human level perfomance, you still have good ways to progress:
# 	> Get labels from humans.
# 	> Error analysis.
# 	> Estimate bias variance effect.

