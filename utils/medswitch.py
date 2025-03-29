import json
import os
import time

import pandas as pd
import regex as re
import numpy as np
import openai
import tiktoken
import json

keys = os.environ["OPENAI_API_KEY"]

from sklearn.model_selection import train_test_split

### OPENAI PARAMS
openai.api_type = "azure"
openai.api_base = keys["OPENAI_API_URL"]
openai.api_version = "2023-05-15" 
openai.api_key = keys["OPENAI_API_KEY"] #os.getenv("OPENAI_API_KEY")

def get_first_or_last_values(data_df,
                              datetime_col="startdatekeyvalue", 
                              get_first=True, 
                              groupby="patientdurablekey",
                             prefix=None):
    '''
    Gets first or last value for each group when sorted by datetime_col 
    '''
    data_df = data_df.sort_values(datetime_col, ascending=get_first) # gets first occurrence if get_first is True
    data_df = data_df.groupby(groupby, sort=False).first()
    if prefix is not None:
        data_df.columns = [str(prefix)+s for s in data_df.columns]
    return data_df

def fill_prev_next_meds(ground_truth_df, med_col = "mapped_med_generic_clean"):
    """
    Helper class for medication table querying to get the previous and next medications
    """
    ## Get ground truth values for medication encounters
    ground_truth_df = ground_truth_df.sort_values(["patientdurablekey", "startdatekeyvalue"])
    ground_truth_df["next_medication"] = ground_truth_df.groupby("patientdurablekey", sort=False)[med_col].shift(periods=-1, freq=None, axis=0)
    ground_truth_df["prev_medication"] = ground_truth_df.groupby("patientdurablekey", sort=False)[med_col].shift(periods=1, freq=None, axis=0)
    ground_truth_df["curr_med_change"] = [False if p is None
                                       else False if type(p)==float
                                       else c!=p for c,p in zip(ground_truth_df[med_col], ground_truth_df["prev_medication"])]
    ground_truth_df["next_med_change"] = [False if n is None
                                       else False if type(n)==float
                                       else c!=n for c,n in zip(ground_truth_df[med_col], ground_truth_df["next_medication"])]
    
    return ground_truth_df

def unique_trajectory(med_trajectory):
    """
    Given a list of medications, get the unique trajectory
    Eg. ["etanercept", "baricitinib", "baricitinib", "etanercept"] -> ["etanercept", "baricitinib", "etanercept"]
    """
    
    return [med_trajectory[i] for i in range(len(med_trajectory)) if (i==0) or med_trajectory[i] != med_trajectory[i-1]]

def _test_unique_trajectory():
    assert unique_trajectory(["etanercept", "baricitinib", "baricitinib", "etanercept"]) == ["etanercept", "baricitinib", "etanercept"]

def cleanNotes(nlp_df):
    """
    Clean notes by removing extra new lines and "*****" -> ""
    """
    nlp_df["note_text_clean"] = [re.sub(r'\n\s*\n', '\n', s) for s in nlp_df["note_text"]]
    nlp_df["note_text_clean"] = [s.replace("*****", "") for s in nlp_df["note_text_clean"]]
    return nlp_df

def preprocess_for_max_length_tiktoken(text, encoder, n):
    """
    Tiktoken backend for max length preprocessing
    
    Parameters:
        text (str): 
        encoder (tiktoken.Encoding): 
        n (int): max length
    
    """
    encoded_texts = encoder.encode(text)
    encoded_texts = [encoded_texts[i:i+n] for i in range(0, len(encoded_texts), n)]
    encoded_texts = [encoder.decode(t) for t in encoded_texts]

    return encoded_texts

def splitNoteTableByMaxLength(med_class_name,
                              model = "gpt-3.5-turbo",
                              max_length = 3700, drop=True):
    """
    Given a set of annotated notes, split the notes and explode the dataframe
    Notes will be indexed by encounterkey_N where N is the index of the note from that encounter
    
    GPT4 has max token length of 8,192 (7750 token max notes)
    gpt-3.5-turbo has max token length of 4096 (3700 token max notes)
    text-davinci-003 has max token length of 4001 (3600 token max notes)
    
    Parameters
    -----------
    med_class_name (str): name of the medication class to use
    drop (bool): drops values longer than max length if true, otherwise, explodes dataset into multiple notes per value
    
    Parameters
    -----------
    
    """
    note_rdd = pd.read_parquet(f"./data/{med_class_name}/{med_class_name}_annotated_notes.parquet.gzip")

    ## Clean and preprocess notes
    # Remove special characters denoting redacted text and extra white lines
    note_rdd = cleanNotes(note_rdd)

    # Get token values
    enc = tiktoken.encoding_for_model(model)
    note_rdd[f"{model}_token_length"] = [len(enc.encode(s)) for s in note_rdd["note_text_clean"]]
    print(note_rdd[f"{model}_token_length"].describe())
    
    # Drop notes that are longer than max length
    if drop:
        return note_rdd[note_rdd[f"{model}_token_length"] > max_length]

    # Split notes based on max length (~150 tokens for prompt and max 150 tokens for response)
    note_rdd["note_text_split"] = note_rdd['note_text_clean'].apply(lambda data: preprocess_for_max_length_tiktoken(data, encoder=enc, n=max_length))

    # keep track of where in note the value is 
    note_rdd["multi_note_index"] = note_rdd['note_text_split'].apply(lambda data: list(range(len(data))))

    # explode
    note_rdd = note_rdd.explode(["note_text_split", "multi_note_index"])

    note_rdd["multi_note_index"] = [e[0]+ "_" +str(m) for e, m in zip(note_rdd.index, note_rdd["multi_note_index"])]
    note_rdd[f"multi_note_{model}_tokens"] = [len(enc.encode(s)) for s in note_rdd["note_text_split"]]
    print(note_rdd[f"multi_note_{model}_tokens"].describe())

    note_rdd.to_parquet(f"./data/{med_class_name}/{med_class_name}_annotated_notes_{model}_split.parquet.gzip", compression="gzip")
    
def map_generic(medication_name, mapping_dict, return_value=False):
    """
    Maps medication name to a generic value based on mapping dict
    Returns None if return_value is False otherwise returns the original value
    """
    keep_original = medication_name
    medication_name = medication_name.lower()
    for k in mapping_dict:
        if k.lower() in medication_name:
            return mapping_dict[k]
        elif mapping_dict[k].lower() in medication_name:
            return mapping_dict[k]
        elif medication_name in k.lower():
            return mapping_dict[k]
        elif medication_name in mapping_dict[k].lower():
            return mapping_dict[k]
    
    if return_value:
        return keep_original
    return None

def savePatientData(med_class_name, 
                    med_name_col="medicationname_generic",
                   med_class_col="medclass_clean"):
    """
    Save patient metadata from list of medication table values
    
    Parameters
    -----------
    med_class_name (str): name of therapeutic class to label files generated by this function
    med_name_col (str): column containing generic name of medication
    med_class_col (str): column containing medication class 
    
    Returns
    -----------
    None
    """
    med_df = pd.read_parquet(f"./data/{med_class_name}/{med_class_name}_data.parquet.gzip")

    ## Get patient medication trajectories & demographics
    med_pts = med_df.groupby(["patientdurablekey"]).first()[[]]
    med_pts = pd.DataFrame(med_pts)

    trajectory_dict = {med_name_col:"med_trajectory", # biosimilars grouped
                       med_class_col:"class_trajectory"}

    for med_col in trajectory_dict:
        traj_med_col = trajectory_dict[med_col]
        med_pts[traj_med_col] = med_df.groupby("patientdurablekey")[med_col].apply(list)
        med_pts[traj_med_col+"_unique"] = [[A[i] for i in range(len(A)) if (i==0) or A[i] != A[i-1]] for A in med_pts[traj_med_col]]
        med_pts[traj_med_col+"_count"] = med_pts[traj_med_col+"_unique"].apply(lambda x: len(x)) # 1 + number of switches

    ## Save all patient demogephics with medication trajectories (all patients taking any contraceptives)
    pt_cols = ["patientdurablekey", "smokingstatus", "primaryfinancialclass", "mychartstatus",
               "ucsfderivedraceethnicity_x", "sex", "sexassignedatbirth", "genderidentity", "postalcode",
              "preferredlanguage", "birthdate", "deathinstant"]
    pt_rdd = load_register_table("DEID_CDW", "patdurabledim", **{"columns":pt_cols})
    pt_rdd = pt_rdd[pt_rdd["patientdurablekey"].isin(list(med_pts.index))]
    pt_rdd = pt_rdd.compute()

    # format and save
    med_pts = pt_rdd.merge(med_pts, left_on="patientdurablekey", right_index=True, how="inner")
    med_pts.to_parquet(f"./data/{med_class_name}/{med_class_name}_pts.parquet.gzip", compression="gzip")

def _manual_json_formatting(text):
    """Extract JSON from response. Returns json (dict) if available, else None"""

    # deal with extra spaces and bad characters
    text = re.sub(r'\t', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub("[ ]*,[ ]*",",", text)
    text = re.sub("[ ]*{[ ]*","{", text)
    text = re.sub("[ ]*:[ ]*",":", text)
    text = re.sub("[ ]*}[ ]*","}", text)
    text = text.replace("\x07", "")
    
    try:
        text = text.split("{", 1)[-1]
        text = text[:text.rindex("}")]
        text = text.strip()
        add_curly_brace = text.count("{") - text.count("}")
        text = text+"}"*add_curly_brace
        text = "{"+text+"}"

        # Deal with ' and " values in text
        #if " is not next to a comma or parenthesis, make it a \'
        #},'
        text = re.sub(r"(?<!([,:}{]))\"(?!([,:}{]))", "\'", text)
        text = re.sub(r"(?<!(\",))(?<=([,]))\"(?!([,:}{]))", "\'", text)
        text = re.sub(r"(?<=([,:}{]))\'(?!([,:}{]))", "\"", text)

        return json.loads(text)
    except:
        return text

def format_gpt_json_response(response):
    """
    Used with prompt that places values in {"json":str, "format":list<str>}"
    """
    # Get prompt information
    query_values = {}
    query_values["prompt_tokens"] = response["usage"]["prompt_tokens"]
    query_values["completion_tokens"] = response["usage"]["completion_tokens"]

    # extract text
    text = response.choices[0]["message"]
    
    if "function_call" in text:  # using function calling  
        text = text["arguments"]
    elif "content" in text: 
        text = _manual_json_formatting(text["content"])

    try:
        query_values.update(text)
        return query_values
    except:
        print("Error converting response to json")
        query_values["json_error"] = text
        return query_values

def clean_gpt_med_values(values, med_clean_dict):
    """
    Clean up names from gpt values
    
    values (list<str>)
    med_clean_dict (dict): values to map
    """
    # get every matching medication
    values = [np.nan if type(c)!=str else c.split(",") for c in values]
    values = [np.nan if type(c)!=list 
              else [med_clean_dict[k] for k in med_clean_dict for med in c if k in med.lower()] # TODO: check if better
              for c in values]
    
    # filter out "none" and "other" values
    values = ["None" if type(c)!=list 
              else "Other" if len(c)==0
              else ",".join(list(set(c))) 
              for c in values]
    
    return values

def run_gpt4_query(notes_df,
                   outfile,
                   engine="gpt-4",
                   temperature=0,
                   max_tokens=400,
                   top_p=1,
                   sys_message=None,
                   task=None,
                   functions=None,
                   **kwargs):
    """
    
    """

    # Query parameters

    if functions is not None:
        kwargs.update({"functions":functions})

    # Querying
    response_dict = {}
    curr_ind=0
    print("SYS MESSAGE:", sys_message)
    print("PROMPT:",task)
    for note_key, note in zip(notes_df["note_deid_note_key"], notes_df["note_text"]):
        
        prompt = f"Clinical note: \"\"\"{note}\"\"\"\n" + task
        try:
            response = openai.ChatCompletion.create(engine=engine,
                                                    messages=[
                                                        {"role": "system", "content": sys_message},
                                                        {"role": "user","content": prompt}
                                                    ],
                                                    temperature=temperature, 
                                                    max_tokens=max_tokens,
                                                    top_p=top_p,
                                                    frequency_penalty=0,
                                                    presence_penalty=0,
                                                    **kwargs
                                                    )
        except:
            time.sleep(50)
            print("time")
            response = openai.ChatCompletion.create(engine=engine,
                                                    messages=[
                                                        {"role": "system", "content": sys_message},
                                                        {"role": "user","content": prompt}
                                                    ],
                                                    temperature=temperature, 
                                                    max_tokens=max_tokens,
                                                    top_p=top_p,
                                                    frequency_penalty=0,
                                                    presence_penalty=0,
                                                    **kwargs
                                                    )
        
        response_clean = format_gpt_json_response(response)
        response_clean["sys_message"] = sys_message
        #response_clean["deid_note_key"] = f"Clinical note: \"\"\"{note_key}\"\"\"\n"
        #response_clean["task"] = task
        response_dict[note_key] = response_clean
        
        if curr_ind%15==0:
            print(note_key)
            responses_df = pd.DataFrame.from_dict(response_dict, orient="index")
            #print(responses_df.filter(regex="new|first|last|reason").tail())
            
            responses_df.to_csv(outfile)

        curr_ind = curr_ind+1

    responses_df = pd.DataFrame.from_dict(response_dict, orient="index")
    #responses_df = responses_df.applymap(lambda x: ",".join(x) if type(x)==list else x)
    responses_df.to_csv(outfile)

def set_sys_message(prompts, config="default"):
     return prompts["sys_config"][config]

def set_task(prompts, config="default"):
    return prompts["task_config"][config]

def _set_functions(config="default"):
    if config=="default":
        functions = [
            {
                "name": "extract_treatment_strategy",
                "description": "Non-conventional disease-modifying anti-rheumatic drugs (ncDMARDs) include biologic drugs or JAK inhibitors.",
                "parameters": {
                    "type": "object",
                        "properties": {
                            "new_ncdmard": {
                                "type": "string",
                                "description": 'What new ncDMARD was discussed, proposed, or prescribed? If the patient is not starting a new tmard, write "NA"'
                            },
                            "last_ncdmard": {
                                "type": "string",
                                "description": 'What was the last ncDMARD the patient used? If none, write "NA"'
                            },
                            "reason_last_ncdmard_stopped": {
                                "type": "string",
                                "description": 'Specific reason(s) why the last ncDMARD was stopped or planned to be stopped. If the ncDMARD was not stopped, write "NA"'
                            },
                        },
                    "required": ["new_ncdmard", "last_ncdmard", "reason_last_ncdmard_stopped"],
                },
            }
        ]
      
    return functions

def split_validation_test(med_class_name, validation_size=0.05, seed=0):
    """
    Splits dataframe into validation ("prompt development") and test sets 
    """
    print(med_class_name)
    ## Load notes
    notes_df = pd.read_parquet(f"./data/{med_class_name}/annotated_medications.parquet.gzip")

    ## Only limit to encounters where there was a medication change
    notes_df = notes_df[notes_df["curr_med_change"]]
    notes_df = notes_df[notes_df["prev_medication"]!=None]

    # Split prompt dev/test set
    pts_list = list(notes_df["patientdurablekey"].unique())

    test_pts, prompt_pts = train_test_split(pts_list, test_size=validation_size, random_state=seed, shuffle=True)
    print("patient split", len(prompt_pts), len(test_pts))
    #test_pts, valid_pts = train_test_split(test_pts, test_size=0.50, random_state=0, shuffle=True)
    #train_df = notes_df[notes_df["patientdurablekey"].isin(train_pts)]
    prompt_dev_df = notes_df[notes_df["patientdurablekey"].isin(prompt_pts)]
    test_df = notes_df[notes_df["patientdurablekey"].isin(test_pts)]
    print(prompt_dev_df.shape)
    print(test_df.shape)

    #train_df.to_parquet(f"./data/{med_class_name}/gpt4/train.parquet.gzip")
    prompt_dev_df.to_parquet(f"./data/{med_class_name}/gpt4/validation.parquet.gzip")
    test_df.to_parquet(f"./data/{med_class_name}/gpt4/test.parquet.gzip")

