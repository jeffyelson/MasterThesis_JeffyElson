import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset,DatasetDict
import shap
import torch
from huggingface_hub import list_models
from transformers_interpret import PairwiseSequenceClassificationExplainer
import streamlit.components.v1 as components
import pandas as pd


st.set_page_config(
    page_title="Explainability",
    page_icon="🧠",
)

def get_user_models(username):
    models = list_models(author=username, fetch_config=True)
    model_names = [model.modelId for model in models]
    return model_names

# Load the model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer(model_name):
    with st.spinner('Loading model...'):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return model, tokenizer

# Load the dataset
@st.cache(allow_output_mutation=True)
def load_hf_dataset(dataset_name, split):
    with st.spinner('Loading dataset...'):
        if dataset_name == "Sentence level attribution":
            dataset = "final_files/dataset_sentenceattribution_nerfeatures_split.csv"
        elif dataset_name == "Semantic level attribution":
            dataset = "final_files/dataset_semanticattribution_nerfeatures_split.csv"
        elif dataset_name == "Proposition level attribution":
            dataset = "final_files/dataset_propositionattribution_nerfeatures.csv"

        if dataset_name == "Proposition level attribution" or dataset_name == "Sentence level attribution":
            dataset = load_dataset('csv', data_files=dataset, delimiter=',',
                               column_names=["claim", "premise", "label", "category", "count_bf", "count_ca", "count_dis",
                                             "count_food", "count_lipid", "count_treat", "pres_bf", "pres_ca", "pres_dis",
                                             "pres_food", "pres_lipid", "pres_treat", "counte_bf", "counte_ca",
                                             "counte_dis", "counte_food", "counte_lipid", "counte_treat", "prese_bf",
                                             "prese_ca", "prese_dis", "prese_food", "prese_lipid", "prese_treat", "url",
                                             "entities", "entity_map", "entity_map_ev", "entity_ev", "gem_exp", "gem_label", "gpt_label", "gpt_exp",
                                             "gold_exp","synonym","voice","split"], skiprows=1)
        elif dataset_name == "Semantic level attribution":
            dataset = load_dataset(
                'csv',
                data_files=dataset,
                delimiter=',',
                column_names=[
                    "claim", "premise", "label", "category", "count_bf", "count_ca", "count_dis",
                    "count_food", "count_lipid", "count_treat", "pres_bf", "pres_ca", "pres_dis",
                    "pres_food", "pres_lipid", "pres_treat", "counte_bf", "counte_ca", "counte_dis",
                    "counte_food", "counte_lipid", "counte_treat", "prese_bf", "prese_ca", "prese_dis",
                    "prese_food", "prese_lipid", "prese_treat", "url", "entities", "entity_map",
                    "gold_exp", "gemini_exp", "gemini_label", "entity_ev", "entity_map_ev", "split"
                ],
                skiprows=1
            )

        train_dataset = dataset['train'].filter(lambda example: example['split'] == 'train')
        validation_dataset = dataset['train'].filter(lambda example: example['split'] == 'validation')
        test_dataset = dataset['train'].filter(lambda example: example['split'] == 'test')
        dataset = DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset,
            'test': test_dataset
        })
        dataset_split = dataset[split]

        return dataset_split

def choose_claim(dataset, target_claim):
    claims = dataset["claim"]
    claim_indexes = []
    for i, claim in enumerate(claims):
        if claim == target_claim:
            claim_indexes.append(i)

    return claim_indexes
# User input for Hugging Face username
hf_username = st.sidebar.text_input("Hugging Face Username", "jeffyelson03")

# Display models in a dropdown
model_names = get_user_models(hf_username)
selected_model = st.sidebar.selectbox("Select a Model", model_names)
st.title("Explaining Model Predictions for MediClaim Dataset")

# Load selected model and dataset
model, tokenizer = load_model_and_tokenizer(selected_model)
dataset_types = [
    "Sentence level attribution",
    "Semantic level attribution",
    "Human attribution",
    "Proposition level attribution"
]
selected_dataset_type = st.sidebar.selectbox("Select Evidence Type", dataset_types)

split_types = ["train", "test", "validation"]
selected_split_type = st.sidebar.selectbox("Select Split Type", split_types)

data = load_hf_dataset(selected_dataset_type, selected_split_type)

selected_claim = st.selectbox("Select a Claim to Analyze", data["claim"])
claim_indexes = choose_claim(data,selected_claim)

pertubations = ["None", "Synonym Replacement", "Swap Active to Passive(vice versa)"]
selected_pertubation_type = st.sidebar.selectbox("Select Input Pertubation Type", pertubations)


if selected_pertubation_type == "None":

    features = ["No features", "Count and Presence NER Features for Claim", "Count and Presence NER Features for Evidence", "Count and Presence NER Features for Both"]
    selected_feature_type = st.sidebar.selectbox("Select Feature Type", features)









def predict_and_explain(data, index, pertubation, model, tokenizer):
    with st.spinner('Generating explanations...'):
        claim = data["claim"][index]
        #evidence = data["premise"][index]
        if pertubation == "Synonym Replacement":
            evidence = data["synonym"][index]
            st.markdown(f"**Original Evidence:** {data['premise'][index]}")
            st.markdown(f"**Pertubated Evidence:** {evidence}")
        elif pertubation == "Swap Active to Passive(vice versa)":
            evidence = data["voice"][index]
            st.markdown(f"**Original Evidence:** {data['premise'][index]}")
            st.markdown(f"**Pertubated Evidence:** {evidence}")
        elif pertubation == "None":
            evidence = data["premise"][index]

            if selected_feature_type == "Count and Presence NER Features for Claim":
                additional_features = [
                    "count_bf", "count_ca", "count_dis", "count_food", "count_lipid", "count_treat", "pres_bf",
                    "pres_ca", "pres_dis", "pres_food", "pres_lipid", "pres_treat"]

                for feature in additional_features:
                    claim += "[SEP]" + str(data[feature][index])
            elif selected_feature_type == "Count and Presence NER Features for Evidence":
                additional_features = [
                    "counte_bf", "counte_ca", "counte_dis", "counte_food", "counte_lipid", "counte_treat", "prese_bf",
                    "prese_ca", "prese_dis", "prese_food", "prese_lipid", "prese_treat"]

                for feature in additional_features:
                    evidence += "[SEP]" + str(data[feature][index])
            elif selected_feature_type == "Count and Presence NER Features for Both":
                additional_features = [
                    "count_bf", "count_ca", "count_dis", "count_food", "count_lipid", "count_treat", "pres_bf",
                    "pres_ca", "pres_dis", "pres_food", "pres_lipid", "pres_treat"]

                for feature in additional_features:
                    claim += "[SEP]" + str(data[feature][index])

                additional_features_evidence = [
                    "counte_bf", "counte_ca", "counte_dis", "counte_food", "counte_lipid", "counte_treat", "prese_bf",
                    "prese_ca", "prese_dis", "prese_food", "prese_lipid", "prese_treat"]

                for feature_ev in additional_features_evidence:
                    evidence += "[SEP]" + str(data[feature_ev][index])


        label = data["label"][index]

        # Pass the tokenized inputs to the explainer
        multiclass_explainer = PairwiseSequenceClassificationExplainer(model=model, tokenizer=tokenizer)

        #sentence = claim + " [SEP] " + evidence

        word_attributions = multiclass_explainer(evidence, claim)

        predicted_label = multiclass_explainer.predicted_class_name
        if predicted_label == "entailment":
            predicted_label = "SUPPORTED"
        elif predicted_label == "contradiction":
            predicted_label = "REFUTED"
        else:
            predicted_label = "NOT ENOUGH INFORMATION"

        html_data = multiclass_explainer.visualize()
        word_attributions = pd.DataFrame(word_attributions, columns=['Word', 'Attribution'])

        return label, predicted_label, word_attributions, html_data



# Streamlit interface

if st.button("Show Explanation"):
    print(selected_claim)
    for index in claim_indexes:
        label, predicted_label, word_attributions, html_data = predict_and_explain(data,index,selected_pertubation_type,model,tokenizer)

        if label == predicted_label:
            result_message = "Correctly Classified 😊"
        else:
            result_message = "Misclassified 😞"

        # Define the formatted string with HTML
        formatted_text = f"""
        <div style='font-family: Arial; font-size: 16px;'>
            <strong>Actual Label:</strong> {label}, <strong>Predicted Label:</strong> {predicted_label}
            <br><span style='color: {"green" if label == predicted_label else "red"};'>{result_message}</span>
        </div>
        """
        #st.write("### Generated Word Attributions")
        #st.table(word_attributions)

        st.write("### Classification")
        st.markdown(formatted_text, unsafe_allow_html=True)

        st.write("### Visualization of Attention Weights")
        raw_html = html_data._repr_html_()

        components.html(raw_html,width=1000, height=1000)

