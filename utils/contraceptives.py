import pandas as pd
import json
import numpy as np
import tiktoken

from utils.dask_cluster import load_register_table, load_cluster
from utils.medswitch import fill_prev_next_meds, get_first_or_last_values, map_generic,unique_trajectory
#from utils.preprocess import cleanNotes, preprocess_for_max_length_tiktoken
from utils.benchmark.metrics import classification_metrics, substring_search_metrics

def getMedications(filepath):
    ## Get all medications labeled as contraceptives
    orders_table = load_register_table("DEID_CDW", "medicationorderfact")
    med_df = orders_table[orders_table["medicationtherapeuticclass"]=="CONTRACEPTIVES"].compute()

    ## Remove non-drug and emergency contraceptives
    non_med_contraceptives = ["DIAPHRAGMS/CERVICAL CAP", "CONDOMS", "PROGESTATIONAL AGENTS", "ESTROGENIC AGENTS"]
    med_df = med_df[~med_df["medicationpharmaceuticalclass"].isin(non_med_contraceptives)]
    emergency = ["Emergency Contraceptive", "Spermicides", "Antineoplastic", "Vaginal pH Modulator"]
    med_df = med_df[~med_df["medicationpharmaceuticalsubclass"].str.contains("|".join(emergency))]

    ## Save mapping
    med_name_to_class = dict(zip(med_df["medicationname"], med_df["medicationpharmaceuticalsubclass"]))
    generic_name_to_class = dict(zip(med_df["medicationgenericname"], med_df["medicationpharmaceuticalsubclass"]))
    # >> Saved to contraceptives.json
    
    ## Save removed values mapping
    non_med_contraceptives = ["DIAPHRAGMS/CERVICAL CAP", "CONDOMS", "PROGESTATIONAL AGENTS", "ESTROGENIC AGENTS"]
    emergency = ["Emergency Contraceptive", "Spermicides", "Antineoplastic", "Vaginal pH Modulator"]
    med_df = orders_table[orders_table["medicationpharmaceuticalclass"].isin(non_med_contraceptives)|
                         orders_table["medicationpharmaceuticalsubclass"].isin(emergency)].compute()
    med_df.groupby(["medicationname", "medicationpharmaceuticalclass", "medicationpharmaceuticalsubclass"]).first()[[]].to_csv("./figures/supplement/TableS2.csv")
    
    ## Query using those names
    med_list = list(med_df["medicationname"].unique()) + list(med_df["medicationgenericname"].unique())
    med_df = orders_table[orders_table["medicationname"].str.contains("|".join(med_list), case=False)|
                         orders_table["medicationgenericname"].str.contains("|".join(med_list), case=False)].compute()
    print("all medications:", med_df.shape)
    
    # map to class
    med_name_to_class = dict(zip(med_df["medicationname"], med_df["medicationpharmaceuticalclass"]))
    generic_name_to_class = dict(zip(med_df["medicationgenericname"], med_df["medicationpharmaceuticalclass"]))
    med_df["med_class"] = [med_name_to_class[m] if m in med_name_to_class
                               else generic_name_to_class[g] if g in generic_name_to_class
                               else None for m,g in zip(med_df["medicationname"], med_df["medicationgenericname"])]

    med_df = med_df[~med_df["med_class"].isna()]
    print("values falling into defined contraceptive category:", med_df.shape)

    ## Remove medications without a start date value
    med_df = med_df[~med_df["startdatekeyvalue"].isna()]
    print("removed no start date:", med_df.shape)

    ## Drop duplicate values
    med_df = med_df.reset_index().sort_values("startdatekeyvalue")
    med_df = med_df.drop_duplicates(subset=["patientdurablekey", "encounterkey", "medicationpharmaceuticalsubclass"])
    print("after dropping duplicates:", med_df.shape)

    # Save all contraceptive data
    med_df.to_parquet(f"{filepath}/contraceptive_medications.parquet.gzip", compression="gzip")

def getDemographics(filepath="./data/contraceptives/raw"):
    """
    Get demographic data with first and last medications added to each patient
    """
    # Load diagnosis and medications/procedures
    med_orders = pd.read_parquet(f"{filepath}/contraceptive_medications.parquet.gzip")

    # Load patient demographics table and filter to patients with at least 1 contraceptive used
    pts_list = list(med_orders["patientdurablekey"].unique())

    pts_df = load_register_table("DEID_CDW", "patdurabledim")
    pts_df = pts_df[pts_df["iscurrent"]==1]
    pts_df = pts_df[pts_df["isvalid"]==1]
    pts_df = pts_df[pts_df["patientdurablekey"].isin(pts_list)]
    pts_df = pts_df.compute()

    # Add first medication information to demographics
    sort_meds = med_orders[["patientdurablekey", "medicationname", "medicationgenericname", "startdatekeyvalue"]]
    first_contraceptive = get_first_or_last_values(sort_meds, datetime_col="startdatekeyvalue",
                                             get_first=True, groupby="patientdurablekey",
                                             prefix = "first_contraceptive")
    pts_df = pts_df.merge(first_contraceptive, left_on="patientdurablekey", right_index=True, how="left")

    # Add last medication information to demographics
    last_contraceptive = get_first_or_last_values(sort_meds, datetime_col="startdatekeyvalue",
                                             get_first=False, groupby="patientdurablekey",
                                             prefix = "last_contraceptive")
    pts_df = pts_df.merge(last_contraceptive, left_on="patientdurablekey", right_index=True, how="left")

    # Get last encounter for each patient
    pts_list = list(pts_df["patientdurablekey"].unique())
    encounter_table = load_register_table("DEID_CDW", "encounterfact")
    encounter_table = encounter_table[encounter_table["patientdurablekey"].isin(pts_list)]
    encounter_table = encounter_table.dropna(subset=["datekeyvalue"])
    encounter_table = encounter_table.sort_values("datekeyvalue")
    encounter_table = encounter_table.groupby("patientdurablekey", sort=False).last()
    pt_encounters = encounter_table.compute()

    # Add last encounter values to patient demographics
    pt_encounters = pt_encounters[["encounterkey", "datekeyvalue"]]
    pt_encounters.columns = ["last_encounterkey", "last_encounter_datekeyvalue"]
    pts_df = pts_df.merge(pt_encounters, left_on="patientdurablekey",
                          right_index=True, how="left")

    print(pts_df["patientdurablekey"].nunique())

    pts_df.to_parquet(f"{filepath}/contraceptive_demographics.parquet.gzip")

def addNotes(filepath="./data/contraceptives"):
    """
    Filter patients and add clinical notes
    """
    # load patient & medication values
    meds_df = pd.read_parquet(f"{filepath}/contraceptive_medications.parquet.gzip")
    pts_df = pd.read_parquet(f"{filepath}/contraceptive_demographics.parquet.gzip")
    print("Initial # of pts:", pts_df.shape)
    print("Initial # of meds orders/procedures:", meds_df.shape)

    # Not implemented: Filter patients above the age of 18 at first contraceptive
    #pts_df["age_at_med_years"] = [d.days / 365 for d in (pts_df["first_contraceptivestartdatekeyvalue"] - pts_df["birthdate"])]
    #pts_df = pts_df[pts_df["age_at_med_years"]>=18]
    #print("# of adult patients:", pts_df.shape)

    # Filter patients to those with a final encounter at least 6 months after diagnosis
    pts_df["first_contraceptive_to_last_encounter_days"] = [d.days for d in (pts_df["last_encounter_datekeyvalue"] - pts_df["first_contraceptivestartdatekeyvalue"])]
    pts_df = pts_df[pts_df["first_contraceptive_to_last_encounter_days"]>180]
    print("# of patients with final encounter >6 months after first contraceptive:", pts_df.shape)

    # Save filtered pts_df
    pts_df.to_parquet(f"{filepath}/final_cohort_pts.parquet.gzip")

    # Filter medications to relevant patients
    meds_df = meds_df[meds_df["patientdurablekey"].isin(pts_df["patientdurablekey"])]
    print("# of medications for new filtered patients:", meds_df.shape)

    # Get clinical notes written by relevant specialists associated with medication encounters
    notes_rdd = load_register_table("DEID_CDW", "note_text")
    notes_meta_rdd = load_register_table("DEID_CDW", "note_metadata")
    med_encounters = set(meds_df["encounterkey"])
    med_notes_meta = notes_meta_rdd[notes_meta_rdd["encounterkey"].isin(list(med_encounters))]

    #prov_specialties=["Obstetrics and Gynecology", "Gynecology", "Women's Health", "Primary Care", "General Internal Medicine"]
    #med_notes_meta = med_notes_meta[med_notes_meta["prov_specialty"].isin(prov_specialties)]

    # Limit to progress/plan notes
    #note_types = "Progress Notes|Plan Note"
    #med_notes_meta = med_notes_meta[med_notes_meta["note_type"].str.contains(note_types, na=False)]

    med_notes_meta = med_notes_meta.merge(notes_rdd, left_on="deid_note_key", right_on="deid_note_key", how="inner")
    med_notes_meta = med_notes_meta.compute()

    # Add clinical notes to medication tables
    med_notes_meta = med_notes_meta[["note_text", "prov_specialty", "encounter_type", "note_type", 
                    "encounterkey", "enc_dept_name", "enc_dept_specialty", "deid_service_date", "deid_note_key"]]
    med_notes_meta.columns = ["note_"+s if not s.startswith("note") else s for s in med_notes_meta.columns]

    meds_df = meds_df.merge(med_notes_meta, left_on="encounterkey", right_on="note_encounterkey", how="left")
    print("# of medications with associated notes:", meds_df.shape)

    meds_df = meds_df[meds_df["startdatekeyvalue"]==meds_df["note_deid_service_date"]]
    print("# of medications with associated notes where note date is same as prescription date:", meds_df.shape)
    print("# of patients with relevant medications & notes:", meds_df["patientdurablekey"].nunique())
    print("# of unique encounters:",  meds_df["encounterkey"].nunique())

    # Save medications associated with annotated notes
    meds_df.to_parquet(f"{filepath}/contraceptive_meds_with_notes.parquet.gzip")
    
def finalWeakAnnotations():
    # Get medication mapping
    with open("./data/contraceptives/raw/contraceptives.json") as map_file:
        med_class_map = json.load(map_file)
        
    med_name_to_class = med_class_map["cdw_medname_pharmaceuticalsubclass"]
    generic_name_to_class = med_class_map["cdw_medgenericname_pharmaceuticalsubclass"]

    # Get medications and notes
    med_notes_df = pd.read_parquet("./data/contraceptives/raw/contraceptive_meds_with_notes.parquet.gzip")
    
    # Map medication names to generic values
    med_notes_df["med_subclass"] = [med_name_to_class[m] if m in med_name_to_class
                                   else generic_name_to_class[g] if g in generic_name_to_class
                                   else None for m,g in zip(med_notes_df["medicationname"], med_notes_df["medicationgenericname"])]
    med_notes_df["mapped_med_generic_clean"] = ["Oral" if "Oral" in m
                                               else "Intrauterine" if "IUD" in m
                                               else "Implant" if "Implant" in m
                                               else "Intravaginal" if "Intravaginal" in m
                                                else "Injectable" if "Injectable" in m
                                                else "Transdermal" if "Transdermal" in m
                                               else None for m in med_notes_df["med_subclass"]]
    
    # Get patient dataframe
    pts_df = pd.read_parquet("./data/contraceptives/raw/final_cohort_pts.parquet.gzip")

    # drop duplicates
    print("Original number of medications with clinical notes:", med_notes_df.shape)
    print("Original number of patients with clinical notes:", med_notes_df["patientdurablekey"].nunique())
    med_notes_df = med_notes_df.drop_duplicates(subset=["patientdurablekey", "mapped_med_generic_clean", "encounterkey", "startdatekeyvalue"], keep="first")
    print("Drop duplicate medications with same class at the same encounter on same date:",med_notes_df.shape)
    print("# patients remaining:", med_notes_df["patientdurablekey"].nunique())

    # Drop encounters that have multiple prescriptions
    multiple_counts = med_notes_df.groupby("encounterkey", sort=False)["mapped_med_generic_clean"].count().sort_values()
    multiple_counts = multiple_counts[multiple_counts>1]
    print(f"Remove {len(multiple_counts)} encounters with multiple contraceptives", )

    med_notes_df = med_notes_df[~med_notes_df["encounterkey"].isin(multiple_counts.index)]
    print("Medication orders remaining:", med_notes_df.shape)
    print("Patients remaining:", med_notes_df["patientdurablekey"].nunique())
    
    # Add token counds and remove notes under 49 tokens
    enc = tiktoken.encoding_for_model("gpt-4")
    med_notes_df["note_tokens"] = [len(enc.encode(s)) for s in list(med_notes_df["note_text"])]
    med_notes_df = med_notes_df[med_notes_df["note_tokens"]>49]
    
    # Create previous and next columns
    med_notes_df = fill_prev_next_meds(med_notes_df)

    # Save annotated medications
    med_notes_df.to_parquet("./data/contraceptives/annotated_medications.parquet.gzip")

    # Add medication trajectory for each patient
    final_med_notes_df = med_notes_df[["patientdurablekey", "medicationname", "encounterkey",
                                       "medicationgenericname",  "mapped_med_generic_clean", 
                                       "startdatekeyvalue", "enddatekeyvalue", "note_deid_note_key"]]
    final_med_notes_df = final_med_notes_df.sort_values("startdatekeyvalue")

    for col in final_med_notes_df.columns:
        med_trajectory = final_med_notes_df.groupby("patientdurablekey", sort=False)[col].apply(list)
        pts_df["final_"+col] = pts_df["patientdurablekey"].map(med_trajectory)

    # Add in unique medication trajectory and labels for patient switching
    pts_df["final_unique_med_trajectory"] = [None if type(t)!=list else unique_trajectory(t) for t in pts_df["final_mapped_med_generic_clean"]]

    pts_df["med_switching_label"] = ["No contraceptive with note" if u is None
                                     else "Contraceptive switch" if len(u)>1
                                     else "No switch" for u in  pts_df["final_unique_med_trajectory"]]
    # Save annotated patient data
    print()
    print("Distribution of treatment switching:")
    print(pts_df["med_switching_label"].value_counts())
    
    pts_df.to_parquet("./data/contraceptives/annotated_pt_demographics.parquet.gzip")

def evaluate_prompt_dev(prompt_dev_df, 
                      med_mapping,
                      date,
                      med_class_name="contraceptives",
                      engine="gpt-4",
                     average="micro"):
    """
    Full eval loop for prompt development dataset
    """
    eval_dfs = {"class":pd.DataFrame(), "text":pd.DataFrame(), "pred_values":pd.DataFrame()}

    #function_config= None # Currently not implemented 
    for sys_config in ["general" ,"specialist", "default"]: #, 
        for task_config in ["manual-function","default"]: # 
            outfile = f"./data/{med_class_name}/gpt4/prompt_dev/{date}_{engine}_{sys_config}-sys-config_{task_config}-task-config_prompt_dev.csv"
            
            gpt4_df = pd.read_csv(outfile, index_col=0)
            
            gpt4_df = gpt4_df.loc[list(prompt_dev_df["note_deid_note_key"])]

            # Classification values
            class_metrics = {}
            pred_values = {}
            for pred_col, label_col in [("new_contraceptive","mapped_med_generic_clean"), ("last_contraceptive","prev_medication")]:
                preds = list(gpt4_df[pred_col])
                preds = [None if type(p)!=str else map_generic(p,
                                                               med_mapping,
                                                               return_value=False) for p in preds]

                preds = ["" if p is None else p for p in preds]
                labels = list(prompt_dev_df[label_col])
                
                class_metrics[label_col] = classification_metrics(preds=preds, labels=labels, average=average)
                pred_values[label_col+"_preds"] = preds
                pred_values[label_col+"_labels"] = labels
        
            # Free text values
            text_metrics={}
            labels = list(prompt_dev_df["note_text"])
            note_ids = list(prompt_dev_df["note_deid_note_key"])
            for pred_col in ["reason_last_contraceptive_stopped"]:
                # Get predictions
                preds = list(gpt4_df[pred_col])
                preds = ["" if type(p)!=str 
                            else "" if p=="NA"
                            else p for p in preds]
                preds = [p.split("Description")[-1].strip(" ':}")
                         if "Description" in p
                         else p.strip(" ':}") for p in preds]

                # Get substring scores
                substring_scores = substring_search_metrics(preds, labels, ignore_empty_preds=True)
                substring_scores = [s/100 for s in substring_scores]
                text_metrics[pred_col] = {"mean":np.mean(substring_scores), 
                                        "std":np.std(substring_scores), 
                                        "median":np.median(substring_scores)}
                
                # Store predictions and labels
                pred_values[pred_col+"_preds"] = preds
                pred_values[pred_col+"_labels"] = note_ids
                pred_values[pred_col+"_score"] = substring_scores

            # Update full dataframe and store current values
            # metric_set = ["class", "text", "pred"]
            for metric_set_name, curr_metrics in zip(eval_dfs, [class_metrics, text_metrics, pred_values]):
                all_class_df = eval_dfs[metric_set_name]
                curr_class_df = pd.DataFrame.from_dict(curr_metrics, orient="index")

                curr_class_df["sys_config"] = sys_config
                curr_class_df["task_config"] = task_config
                all_class_df = pd.concat([all_class_df, curr_class_df])
                eval_dfs[metric_set_name] = all_class_df
    
    #return eval_dfs
    eval_dfs["text"].to_csv(f"./data/{med_class_name}/gpt4/eval/prompt_dev_text_metrics_{average}.csv")
    eval_dfs["class"].to_csv(f"./data/{med_class_name}/gpt4/eval/prompt_dev_classification_metrics_{average}.csv")
    eval_dfs["pred_values"].to_csv(f"./data/{med_class_name}/gpt4/eval/prompt_dev_evaluated_preds.csv")

def evaluate_test(test_labels_df, 
                      med_mapping,
                      date,
                  sys_config="specialist",
                  task_config="default",
                      med_class_name="contraceptives",
                      engine="gpt-4",
                     average="micro"):
    """
    Full eval loop for prompt development dataset (GPT4)
    """
    eval_dfs = {"class":pd.DataFrame(), "text":pd.DataFrame(), "pred_values":pd.DataFrame()}

    #function_config= None # Currently not implemented 
    outfile = f"./data/{med_class_name}/gpt4/{date}_{engine}_{sys_config}-sys-config_{task_config}-task-config_test.csv"
    
    gpt4_df = pd.read_csv(outfile, index_col=0)
    gpt4_df = gpt4_df.loc[list(test_labels_df["note_deid_note_key"])]
    
    # Classification values
    class_metrics = {}
    pred_values = {}
    for pred_col, label_col in [("new_contraceptive","mapped_med_generic_clean"), ("last_contraceptive","prev_medication")]:
        preds = list(gpt4_df[pred_col])
        preds = [None if type(p)!=str else map_generic(p,
                                                       med_mapping,
                                                       return_value=False) for p in preds]

        preds = ["" if p is None else p for p in preds]
        labels = list(test_labels_df[label_col])

        class_metrics[label_col] = classification_metrics(preds=preds, labels=labels, average=average)
        pred_values[label_col+"_preds"] = preds
        pred_values[label_col+"_labels"] = labels

    # Free text values
    text_metrics={}
    labels = list(test_labels_df["note_text"])
    note_ids = list(test_labels_df["note_deid_note_key"])
    for pred_col in ["reason_last_contraceptive_stopped"]:
        # Get predictions
        preds = list(gpt4_df[pred_col])
        preds = ["" if type(p)!=str 
                    else "" if p=="NA"
                    else p for p in preds]
        preds = [p.split("Description")[-1].strip(" ':}")
                 if "Description" in p
                 else p.strip(" ':}") for p in preds]

        # Get substring scores
        substring_scores = substring_search_metrics(preds, labels, ignore_empty_preds=True)
        substring_scores = [s/100 for s in substring_scores]
        text_metrics[pred_col] = {"mean":np.mean(substring_scores), 
                                "std":np.std(substring_scores), 
                                "median":np.median(substring_scores)}

        # Store predictions and labels
        pred_values[pred_col+"_preds"] = preds
        pred_values[pred_col+"_labels"] = note_ids
        pred_values[pred_col+"_score"] = substring_scores

    # Update full dataframe and store current values
    # metric_set = ["class", "text", "pred"]
    for metric_set_name, curr_metrics in zip(eval_dfs, [class_metrics, text_metrics, pred_values]):
        all_class_df = eval_dfs[metric_set_name]
        curr_class_df = pd.DataFrame.from_dict(curr_metrics, orient="index")

        curr_class_df["sys_config"] = sys_config
        curr_class_df["task_config"] = task_config
        all_class_df = pd.concat([all_class_df, curr_class_df])
        eval_dfs[metric_set_name] = all_class_df

    #return eval_dfs
    eval_dfs["text"].to_csv(f"./data/{med_class_name}/gpt4/eval/test_text_metrics_{average}.csv")
    eval_dfs["class"].to_csv(f"./data/{med_class_name}/gpt4/eval/test_classification_metrics_{average}.csv")
    eval_dfs["pred_values"].to_csv(f"./data/{med_class_name}/gpt4/eval/test_evaluated_preds.csv")
    