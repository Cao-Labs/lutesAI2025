from dif_description import similarity_score

# Example descriptions (replace these with your real generated and reference descriptions)
generated_description = "Protein acts as a transporter of ions."
reference_description = "This protein functions in ion transport across membranes."

score = similarity_score(generated_description, reference_description)
print(f"Similarity score: {score}")
