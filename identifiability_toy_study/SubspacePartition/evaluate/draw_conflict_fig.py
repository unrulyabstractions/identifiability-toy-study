import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
import re

result_path = "../trainedRs/qwen2.5-merge-bright_cloud-model_name-qwen2.5-metric-euclidean-unit_size-64-search_steps-25-data_source-minipile-merge_thr-0.02-merge_start-20000-merge_interval-8000-max_steps-100000/eval_log_nobel_prize_winner-Field.txt"
layer = "\(11, 'mid'\)"

# result_path = "../trainedRs/gemma2-merge-swift_galaxy-model_name-gemma2-metric-euclidean-unit_size-72-search_steps-25-data_source-minipile-merge_thr-0.01-merge_start-10000-merge_interval-4000-max_steps-150000/eval_log_nobel_prize_winner-Field.txt"
# layer = "\(9, 'mid'\)"

data = OrderedDict()
key = None
with open(result_path) as f:
    for line in f:
        print(line)
        if match := re.search(r"original model behavior \tcontext rate: (0\.\d+) \tparam rate: (0\.\d+)", line):
            orig_context_rate = float(match.group(1))
            orig_param_rate = float(match.group(2))
            data["Clean Run"] = (orig_context_rate, orig_param_rate, orig_context_rate, orig_param_rate)
        elif match := re.search(fr"layer {layer} whole space \(d=(\d+)\):", line):
            dim = match.group(1)
            key = f"$d={dim}$"
        elif match := re.search(fr"layer {layer} subspace (\d+) \(d=(\d+)\):", line):
            index, dim = match.groups()
            key = f"$d_{{{index}}}={dim}$"
        elif match := re.search(r"corrupt param behavior \tcontext rate: ([01]\.\d+) \tparam rate: (0\.\d+)", line):
            corp_param_context_rate = float(match.group(1))
            corp_param_param_rate = float(match.group(2))
        elif match := re.search(r"corrupt context behavior \tcontext rate: (0\.\d+) \tparam rate: ([01]\.\d+)", line):
            corp_context_context_rate = float(match.group(1))
            corp_context_param_rate = float(match.group(2))
            if key is None:
                continue
            elif key in data:
                break
            else:
                data[key] = (corp_param_context_rate, corp_param_param_rate, corp_context_context_rate, corp_context_param_rate)

print(data)

top_data = []
bottom_data = []
x_label = []
for key in data:
    corp_param_context_rate, corp_param_param_rate, corp_context_context_rate, corp_context_param_rate = data[key]
    top_data.append((corp_param_context_rate, 1-corp_param_context_rate-corp_param_param_rate, corp_param_param_rate))
    bottom_data.append((corp_context_context_rate, 1-corp_context_context_rate-corp_context_param_rate, corp_context_param_rate))
    x_label.append(key)

top1, top2, top3 = zip(*top_data)
bot1, bot2, bot3 = zip(*bottom_data)
x = [0, 2.0] + [i + 2.0 for i in range(2, len(top_data))] 
widths = [1.2, 1.2] + [0.6] * (len(top_data) - 2)

top2_bottoms = top1
top3_bottoms = [a + b for a, b in zip(top1, top2)]
bot2_bottoms = bot1
bot3_bottoms = [a + b for a, b in zip(bot1, bot2)]


# colors = ["#4C72B0", "#55A868", "#C44E52"]
colors = ["#82B0D2", "#FFBE7A", "#FA7F6F"]
# colors = ["#2878B5", "#FFBE7A", "#C82423"] 
# colors = ["#9AC9DB", "#F8AC8C", "#FF8884"] 
labels = ["context ans", "others", "parametric ans"]

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(8, 5), dpi=200,
    gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.35}
)
labeled = False
def stacked_bar(ax, x_vals, part1, part2, part3, bottoms1, bottoms2, color1, color2, color3, widths):
    for i in range(len(x_vals)):
        global labeled
        if not labeled:
            label1, label2, label3 = labels
            labeled = True
        else:
            label1, label2, label3 = "", "", ""
        ax.bar(x_vals[i], part1[i], color=color1, width=widths[i], label=label1)
        ax.bar(x_vals[i], part2[i], bottom=bottoms1[i], color=color2, width=widths[i], label=label2)
        ax.bar(x_vals[i], part3[i], bottom=bottoms2[i], color=color3, width=widths[i], label=label3)

stacked_bar(ax_top, x, top1, top2, top3, top2_bottoms, top3_bottoms, *colors, widths)
ax_top.set_ylim(0, 1)
ax_top.set_ylabel("Proportion")
ax_top.set_title("Param Corrupt", pad=10)
ax_top.set_xticks(x)
ax_top.set_xticklabels(x_label, rotation=45, fontsize=8)
ax_top.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
ax_top.axhline(y=data["Clean Run"][0], color='red', linestyle='--', linewidth=1)

stacked_bar(ax_bot, x, bot1, bot2, bot3, bot2_bottoms, bot3_bottoms, *colors, widths)
ax_bot.set_ylim(0, 1)
ax_bot.set_ylabel("Proportion")
ax_bot.set_title("Context Corrupt", pad=10, y=-0.25)
ax_bot.axhline(y=1-data["Clean Run"][1], color='red', linestyle='--', linewidth=1)

ax_bot.set_xticklabels([])
ax_bot.set_xlabel("")

plt.rcParams['figure.dpi'] = 10
plt.subplots_adjust(right=0.75)
plt.tight_layout()
plt.show()