import pandas as pd
import ast
import openai
import os

# Use this function if iterating through individual errors within a base system prompt.
# Inputs: report (initial report to generate errors from), client (openAI client), error_added (individual error statement to test)
# Returns: new error-filled report
def add_iterative_error(report, client, error_added):
    # change this if you want to change the base system prompt when testing individual errors
    system_prompt = "You will be given a radiology report of a chest X-ray. Your task is to change some of the statements in the report so that the report is still clinically plausible but has a different meaning than the previous report. Keep track of the sentence indexes corresponding to the sentences you change in a report. Please also add " + error_added + "\nFor a given report, return a new report with one or more changed sentences according to the above paragraph, a new line, and then a Python dictionary in the following format: {error sentence index : label, explanation, original sentence index]}. Make sure this format is followed exactly, including the spacing. The label is determined by the following:\n0: unchanged sentence\n1: changed sentence\n‘explanation’ is determined by the following:\nWhen the label is 1: 'explanation' should contain one statement about the change made in the sentence. The length of the statement should not exceed 15 words. When the 'label' is 0: 'explanation' should contain 'not applicable'. The error sentence indices should match the total number of sentences of the error report, after any additions, and include an entry for every sentence in the error report. For example, an error report of three sentences should have a dictionary of {0: [label, explanation, original sentence index], 1: [label, explanation, original sentence index], 2: [label, explanation, original sentence index]}. The original sentence index refers to the sentence  index of the original report that was changed. If the error is not based on a sentence from the original report (e.g. an addition) or the sentence is correct, leave the original sentence index blank."
    response = client.chat.completions.create(
    model="gpt-4",
    messages=[
      {"role": "system", "content": f"{system_prompt}"},
      {"role": "user", "content": f"{report}"},
      ],
    max_tokens=400
    )
    return response.choices[0].message.content

# Use this function when generating errors based on entire system prompt.
# Inputs: report (initial report to generate errors from), client (openAI client), system_prompt (GPT error generating prompt)
# Returns: new error-filled report
def add_general_error(report, client, system_prompt):
    response = client.chat.completions.create(
    model="gpt-4",
    messages=[
      {"role": "system", "content": f"{system_prompt}"},
      {"role": "user", "content": f"{report}"},
      ],
    max_tokens=400
    )
    return response.choices[0].message.content

# Generate dataframe tracking study/subject ids, original/error reports, and sentence labeling
# Inputs: mimic_df (subset of mimic used), api_key, prompt (GPT error generating prompt- either an individual error or entire system prompt), iterative (whether or not you are iteratively testing errors)
# Returns: error dataframe
def generate_error_df(mimic_df, api_key, prompt, iterative):
    client = openai.OpenAI(api_key=api_key)
    study_ids, subject_ids, original_reports, error_reports, sentence_labelings = [],[],[],[],[]
    for i in range(len(mimic_df)):
        study_ids.append(mimic_df['study_id'][i])
        subject_ids.append(mimic_df['subject_id'][i])
        cur_report = mimic_df['findings'][i] + mimic_df['impression'][i]
        original_reports.append(cur_report)
        if iterative:
            error_output = add_iterative_error(cur_report, client, prompt)
        else:
            error_output = add_general_error(cur_report, client, prompt)
        # current prompting method assumes that there will be a new line between the error report and labeling dictionary
        error_report, sentence_label  = error_output.splitlines()[0], error_output.splitlines()[2]
        error_reports.append(error_report)
        sentence_labelings.append(sentence_label)

    error_df = pd.DataFrame(
    {'Study ID': study_ids,
     'Subject ID': subject_ids,
     'Original Report': original_reports,
     'Error Report': error_reports,
     'Sentence Labelings': sentence_labelings,
    })
    return error_df

# Splice sentences of generated reports and label them
# Input: error_df (formatted in the generate_error_df function)
# Returns: spliced error sentence dataframe
def splice_sentences(error_df):
    study_ids, subject_ids, error_sentences, labels, error_classes, sequences, ground_truth_sentences = [],[],[],[],[],[],[]
    for i in range(len(error_df)):
        try:
            error_report = error_df['Error Report'][i]
            ground_truth_report = error_df['Original Report'][i]
            spliced_error_report = error_report.split('.')
            spliced_error_report = [i for i in spliced_error_report if len(i.strip())>0]
            spliced_gt_report = ground_truth_report.split('.')
            spliced_gt_report = [i for i in spliced_gt_report if len(i.strip())>0]
            sentence_label = ast.literal_eval(error_df['Sentence Labelings'][i])
            for j in range(len(spliced_error_report)):
                if j in sentence_label.keys():
                    error_classes.append(sentence_label[j][1])
                    labels.append(int(sentence_label[j][0]))
                    if len(str(sentence_label[j][2]).strip())>0:
                        ground_truth_sentences.append(spliced_gt_report[int(sentence_label[j][2])])
                    else:
                        ground_truth_sentences.append("")
                    error_sentences.append(spliced_error_report[j])
                    sequences.append(j)
                    study_ids.append(error_df['Study ID'][i])
                    subject_ids.append(error_df['Subject ID'][i])
        except:
            continue

    spliced_error_df = pd.DataFrame(
    {'Study ID': study_ids,
     'Subject ID': subject_ids,
     'Original Sentence': ground_truth_sentences,
     'Error Sentence': error_sentences,
     'Sequence': sequences,
     'Label': labels,
     'Error Class': error_classes,
    })
    return spliced_error_df

# Iterate through dictionary of errors to try
# Input: error_dict (keys are numbers, values are individual errors to try), mimic_df, api_key, root_path (root path to save results)
# Returns: None
def save_iterative_errors(error_dict, mimic_df, api_key, root_path):
    for i in range(len(error_dict)):
        if not os.path.exists(root_path + f'/{i+1}/'):
            os.mkdir(root_path + f'/{i+1}/')
        error_df = generate_error_df(mimic_df, api_key, error_dict[i], True)
        error_df.to_csv(root_path + f'/{i+1}/error_reports_{i+1}.csv')
        spliced_df = splice_sentences(error_df)
        spliced_df.to_csv(root_path + f'/{i+1}/error_spliced_sentences_{i+1}.csv')
        with open(root_path + '/error_dict.txt', 'w') as file:
            file.write(str(i) + ": " + str(error_dict[i]))

