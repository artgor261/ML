from collections import Counter

arr = list()
cnt = Counter(df["rubrics_id"])

for index, row in df.iterrows():
    if cnt[row["rubrics_id"]] > 100:
        arr.append(row["rubrics_id"])
    else:
        arr.append("other")
df.insert(len(df.columns), "modified_rubrics", arr)
