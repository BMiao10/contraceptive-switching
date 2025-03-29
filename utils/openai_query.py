### IMPORTS
import os
import glob
import json
import pandas as pd

from utils import medswitch

def prompt_dev(med_class_name, 
               engine,
               date,
               sys_config_values = [ "specialist", "crc",  "default","general",],
               task_config_values = ["manual-function", "sbs", "default",],
               function_config=None,  # Currently not implemented 
              ):
    """
    OpenAI model inference on medication switching test set.

    Args:
        med_class_name (str): Name of the medical class for which the test set is being generated.
        date (str): The date to be used in the output filename.
        engine (str, optional): The OpenAI engine to be used for the GPT-4 query. Defaults to "gpt4".
        sys_config_values (str, optional): All system messages to try for the GPT-4 query.
        task_config_values (str, optional): All task configurations (prompts) to use for the GPT-4 query.
        med_mapping (dict): mapping of gpt4 values to generic values for evaluation
        average (str): averaging method for evaluation, defaults to "micro"

    Reads:
        - Prompt configurations from: ./data/{med_class_name}/gpt4/prompt_configs.json
        - Validation dataset from: ./data/{med_class_name}/gpt4/validation.parquet.gzip

    Writes:
        - Results to: ./data/{med_class_name}/gpt4/{date}_{engine}_{sys_config}-sys-config_{task_config}-task-config_prompt_dev.csv
    """
    prompt_dev_df = pd.read_parquet(f"./data/{med_class_name}/gpt4/validation.parquet.gzip")

    # Prompt development 
    with open(f"./data/{med_class_name}/gpt4/prompt_configs.json") as f:
         prompts = json.load(f)

    function_config= None # Currently not implemented 
    for sys_config in sys_config_values: 
        for task_config in task_config_values:
            sys_message = medswitch.set_sys_message(prompts, sys_config)
            task = medswitch.set_task(prompts, task_config)
            #functions = _set_functions(config=function_config) if function_config is not None else None

            outfile = f"./data/{med_class_name}/gpt4/prompt_dev/{date}_{engine}_{sys_config}-sys-config_{task_config}-task-config_prompt_dev.csv"
            print(outfile)
            medswitch.run_gpt4_query(prompt_dev_df,
                    outfile,
                    engine=engine,
                    max_tokens=500,
                    top_p=1,
                    sys_message=sys_message,
                    task=task,
                    functions=None)

def gpt4_test_set(med_class_name,
                  date,
                  engine="gpt4",
                    sys_config = "default",
                    task_config = "manual-function"):
    """
    OpenAI model inference on medication switching test set.

    Args:
        med_class_name (str): Name of the medical class for which the test set is being generated.
        date (str): The date to be used in the output filename.
        engine (str, optional): The OpenAI engine to be used for the GPT-4 query. Defaults to "gpt4".
        sys_config (str, optional): The system configuration to use for the GPT-4 query.
            Defaults to "default".
        task_config (str, optional): The task configuration to use for the GPT-4 query.
            Defaults to "manual-function".

    Reads:
        - Prompt configurations from: ./data/{med_class_name}/gpt4/prompt_configs.json
        - Test dataset from: ./data/{med_class_name}/gpt4/test.parquet.gzip

    Writes:
        - Results to: ./data/{med_class_name}/gpt4/{date}_{engine}_{sys_config}-sys-config_{task_config}-task-config_test.csv

    """
    with open(f"./data/{med_class_name}/gpt4/prompt_configs.json") as f:
        prompts = json.load(f)

    test_df = pd.read_parquet(f"./data/{med_class_name}/gpt4/test.parquet.gzip")
    print(test_df.shape) #2820, 24

    outfile = f"./data/{med_class_name}/gpt4/{date}_{engine}_{sys_config}-sys-config_{task_config}-task-config_test.csv"
    print(outfile)

    medswitch.run_gpt4_query(test_df,
            outfile,
            engine=engine,
            max_tokens=1000,
            top_p=1,
            sys_message=medswitch.set_sys_message(prompts, sys_config),
            task=medswitch.set_task(prompts, task_config),
            functions=None)
