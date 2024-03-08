import os
import json
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    df = pd.read_table(args.annotation_file)
    right_num = 0
    cur_df = df.copy()
    cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
    cur_df.insert(6, 'prediction', None)
    for pred in open(os.path.join(args.result_dir, f"{args.experiment}.jsonl")):
        pred = json.loads(pred)
        cur_df.loc[df['index'] == pred['question_id'], 'prediction'] = pred['text']

    cur_df.to_excel(os.path.join(args.upload_dir, f"{args.experiment}.xlsx"), index=False, engine='openpyxl')

    df = pd.read_excel(os.path.join(args.upload_dir, f"{args.experiment}.xlsx"))

    gt_answers = df.iloc[:,[7]].values
    pred = df.iloc[:,[6]].values
    right_num = 0
    lenth = pred.shape[0]
    for index in range(lenth):
        if pred[index,0] == gt_answers[index,0]:
            right_num += 1
    acc = right_num / lenth
    print(f'准确率为：{acc}')