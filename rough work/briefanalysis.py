import pandas as pd 

df = pd.read_csv("combined.csv")

#add partcipant id 
n_participants = 16
trials_per_participant = len(df) // n_participants 
df["participant_id"] = df.index // trials_per_participant + 1 

#analysis for mean rt
print ("Mean RT:")
print(df.groupby("type")["rt"].mean())
#results are 477.41 for congruent; 571.28 for incongruent

#analysis for accurary 
print("\nAccuracy:")
print(df.groupby("type")["accuracy"].mean())
#accuracy for congruent is 97% and for incongruent is 80%

