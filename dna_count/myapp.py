#############################
# Import libaries
#############################

import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image

#############################
# Page Title
#############################

st.write("""
# DNA Nucleotide Count Web App

This app counts the nucleotide composition of query DNA

***
""")

#############################
# Input Text Box
#############################

st.header("Enter DNA sequence")
st.write(
    "DNA sequences consist of four bases: adenine(A), guanine(G), cytosine(C), and thymine(T). So just generate a DNA sequence using those 4 letters, or use this [DNA Sequence Generator](http://www.faculty.ucr.edu/~mmaduro/random.htm)."
)

sequence_input = ">DNA Query\nGGGGGGGAACGCTGAAGATCTCTTCTTCTCATGACTGAACTCGCGAGGGTCGTGATGTCGGTTCCTTCAAAGGTTAAAGAACAAAGGCTTACTGTGCGCA"

sequence = st.text_area("Sequence input", sequence_input, height=250)
sequence = sequence.splitlines()
sequence = sequence[1:]  # exclude the sequence name
sequence = "".join(sequence)  # concatenate dna sequence all together

st.write("""
***
""")

st.header("INPUT (DNA Query)")

sequence


# DNA nucleotide count
def DNA_nucleotide_count(seq):
    d = {
        "A": seq.count('A'),
        "T": seq.count('T'),
        "C": seq.count('C'),
        "G": seq.count('G')
    }

    return d


X = DNA_nucleotide_count(sequence)

X_label = list(X)
X_values = list(X.values())

nucleotide_dict = {
    "A": "Adenine(A)",
    "C": "Cytosine(C)",
    "G": "Guanine(G)",
    "T": "Thymine(T)"
}

# Display base counts in a df
df = pd.DataFrame.from_dict(X, orient='index')
df = df.rename({0: 'count'}, axis='columns')
df.reset_index(inplace=True)
df = df.rename(columns={"index": "nucleotide"})

# df["nucleotide"] = df.apply(lambda x: (nucleotide_dict[x["nucleotide"]]))
df = df.replace({"nucleotide": nucleotide_dict})

st.subheader("Bases Count")

st.dataframe(df)

# Display bar chart using Altair

st.subheader("Chart Display")

p = alt.Chart(df).mark_bar().encode(x="nucleotide", y="count")

p = p.properties(width=alt.Step(80))

st.write(p)
