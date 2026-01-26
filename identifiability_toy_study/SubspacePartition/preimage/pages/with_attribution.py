import streamlit as st
import os
import random
import time
import re
import pickle
from copy import deepcopy
from glob import glob
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
from utils import more_readable_gpt2_tokens

gpt2_byte_decoder = {'!': 33, '"': 34, '#': 35, '$': 36, '%': 37, '&': 38, "'": 39, '(': 40, ')': 41, '*': 42, '+': 43, ',': 44, '-': 45, '.': 46, '/': 47, '0': 48, '1': 49, '2': 50, '3': 51, '4': 52, '5': 53, '6': 54, '7': 55, '8': 56, '9': 57, ':': 58, ';': 59, '<': 60, '=': 61, '>': 62, '?': 63, '@': 64, 'A': 65, 'B': 66, 'C': 67, 'D': 68, 'E': 69, 'F': 70, 'G': 71, 'H': 72, 'I': 73, 'J': 74, 'K': 75, 'L': 76, 'M': 77, 'N': 78, 'O': 79, 'P': 80, 'Q': 81, 'R': 82, 'S': 83, 'T': 84, 'U': 85, 'V': 86, 'W': 87, 'X': 88, 'Y': 89, 'Z': 90, '[': 91, '\\': 92, ']': 93, '^': 94, '_': 95, '`': 96, 'a': 97, 'b': 98, 'c': 99, 'd': 100, 'e': 101, 'f': 102, 'g': 103, 'h': 104, 'i': 105, 'j': 106, 'k': 107, 'l': 108, 'm': 109, 'n': 110, 'o': 111, 'p': 112, 'q': 113, 'r': 114, 's': 115, 't': 116, 'u': 117, 'v': 118, 'w': 119, 'x': 120, 'y': 121, 'z': 122, '{': 123, '|': 124, '}': 125, '~': 126, '¡': 161, '¢': 162, '£': 163, '¤': 164, '¥': 165, '¦': 166, '§': 167, '¨': 168, '©': 169, 'ª': 170, '«': 171, '¬': 172, '®': 174, '¯': 175, '°': 176, '±': 177, '²': 178, '³': 179, '´': 180, 'µ': 181, '¶': 182, '·': 183, '¸': 184, '¹': 185, 'º': 186, '»': 187, '¼': 188, '½': 189, '¾': 190, '¿': 191, 'À': 192, 'Á': 193, 'Â': 194, 'Ã': 195, 'Ä': 196, 'Å': 197, 'Æ': 198, 'Ç': 199, 'È': 200, 'É': 201, 'Ê': 202, 'Ë': 203, 'Ì': 204, 'Í': 205, 'Î': 206, 'Ï': 207, 'Ð': 208, 'Ñ': 209, 'Ò': 210, 'Ó': 211, 'Ô': 212, 'Õ': 213, 'Ö': 214, '×': 215, 'Ø': 216, 'Ù': 217, 'Ú': 218, 'Û': 219, 'Ü': 220, 'Ý': 221, 'Þ': 222, 'ß': 223, 'à': 224, 'á': 225, 'â': 226, 'ã': 227, 'ä': 228, 'å': 229, 'æ': 230, 'ç': 231, 'è': 232, 'é': 233, 'ê': 234, 'ë': 235, 'ì': 236, 'í': 237, 'î': 238, 'ï': 239, 'ð': 240, 'ñ': 241, 'ò': 242, 'ó': 243, 'ô': 244, 'õ': 245, 'ö': 246, '÷': 247, 'ø': 248, 'ù': 249, 'ú': 250, 'û': 251, 'ü': 252, 'ý': 253, 'þ': 254, 'ÿ': 255, 'Ā': 0, 'ā': 1, 'Ă': 2, 'ă': 3, 'Ą': 4, 'ą': 5, 'Ć': 6, 'ć': 7, 'Ĉ': 8, 'ĉ': 9, 'Ċ': 10, 'ċ': 11, 'Č': 12, 'č': 13, 'Ď': 14, 'ď': 15, 'Đ': 16, 'đ': 17, 'Ē': 18, 'ē': 19, 'Ĕ': 20, 'ĕ': 21, 'Ė': 22, 'ė': 23, 'Ę': 24, 'ę': 25, 'Ě': 26, 'ě': 27, 'Ĝ': 28, 'ĝ': 29, 'Ğ': 30, 'ğ': 31, 'Ġ': 32, 'ġ': 127, 'Ģ': 128, 'ģ': 129, 'Ĥ': 130, 'ĥ': 131, 'Ħ': 132, 'ħ': 133, 'Ĩ': 134, 'ĩ': 135, 'Ī': 136, 'ī': 137, 'Ĭ': 138, 'ĭ': 139, 'Į': 140, 'į': 141, 'İ': 142, 'ı': 143, 'Ĳ': 144, 'ĳ': 145, 'Ĵ': 146, 'ĵ': 147, 'Ķ': 148, 'ķ': 149, 'ĸ': 150, 'Ĺ': 151, 'ĺ': 152, 'Ļ': 153, 'ļ': 154, 'Ľ': 155, 'ľ': 156, 'Ŀ': 157, 'ŀ': 158, 'Ł': 159, 'ł': 160, 'Ń': 173}


st.set_page_config(layout="wide")

for key in ["sel_act_site", "sel_subspace"]:
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]

def change_key_value(key, options, increment):
    next_i = (options.index(st.session_state[key]) + increment) % len(options)
    st.session_state[key] = options[next_i]

def span_maker(token: str, mark: bool = False):
    if mark:
        token = "<u><b>" + token + "</b></u>"
    return '<span>' + token + '</span>' 

def span_maker_attr(token: str, color_value: float, norm_term: float, mark: bool = False):
    if color_value/norm_term > 0.6:
        txt_color = "white"
    else:
        txt_color = "black"
    hover_text = 'title="%.3f"'%color_value
    if mark:
        token = "<u><b>" + token + "</b></u>"
    return '<span style="background-color:rgba(40,116,166,%.2f); color: %s" %s>'%(min(1.0, color_value/norm_term), txt_color, hover_text) + token + '</span>' 

def process_special_tokens(tokens):
    special_tokens = {"&": "&amp;", '"': "&quot;", "'": "&#39;", " ": "&nbsp;"}   # now replace whole word, might need to replace every occurrence in each token
    # return ["&nbsp;" if t == " " else t for t in tokens] "<": "&lt;", ">": "&gt;", 
    return [special_tokens.get(t, t.replace("\n", "\\n").replace("<", "&lt;").replace(">", "&gt;")) for t in tokens]


preimage_path = Path("..") / "visualizations" / f"preimage-{st.session_state.exp_name}"
model_name_site_name = [d.split("-") for d in os.listdir(preimage_path)]
if len(set(n[0] for n in model_name_site_name)) == 1:
    model_name = model_name_site_name[0][0]
    all_sites = sorted([n[1] for n in model_name_site_name])
else:
    raise NotImplementedError
cosine = True   # TODO add code for euclidean in cache_attribution.py

with st.sidebar:
    # st.text(st.session_state.exp_name)

    if "sel_act_site" not in st.session_state:
        st.session_state.sel_act_site = all_sites[0]
    sel_act_site = st.selectbox("choose act site", all_sites, key="sel_act_site")

    subspaces = glob("*-attr", root_dir=preimage_path / f"{model_name}-{sel_act_site}")
    subspaces.sort(key=lambda x: int(x.split("-")[0]))
    subspaces = [s[:-5] for s in subspaces]

    if "sel_subspace" not in st.session_state or st.session_state.sel_subspace not in subspaces:
        st.session_state.sel_subspace = subspaces[0]
    sel_subspace = st.selectbox("choose subspace", subspaces, key="sel_subspace")
    cols = st.columns(2)
    cols[0].button("prev", on_click=change_key_value, args=("sel_subspace", subspaces, -1))
    cols[1].button("next", on_click=change_key_value, args=("sel_subspace", subspaces, 1))

    
    query_ids = os.listdir(preimage_path / f"{model_name}-{sel_act_site}" / f"{sel_subspace}-attr")
    query_ids = [int(i[:-5]) for i in query_ids]
    sel_query_id = st.selectbox("query act idx", query_ids, key="sel_query_id")
    cols = st.columns(2)
    cols[0].button("prev", type="primary", on_click=change_key_value, args=("sel_query_id", query_ids, -1))
    cols[1].button("next", type="primary", on_click=change_key_value, args=("sel_query_id", query_ids, 1))

    with open(preimage_path / f"{model_name}-{sel_act_site}" / f"{sel_subspace}-attr" / f"{sel_query_id}.json") as f:
        query_obj = json.load(f)    # {"query_info": (str_tokens, pos_idx, norm), "preimage": []}

    num_show = st.slider("num sample shown", min_value=1, max_value=20, value=20)
    prev_ctx = st.select_slider("# prev context per sample", ["10", "25", "100", "inf"], value="10")
    futr_ctx = st.select_slider("# future context per sample", ["0", "5", "25", "inf"], value="5")

    show_norm = st.toggle("show norm")
    if len(query_obj["query_info"]) == 3:
        query_obj["query_info"] = query_obj["query_info"] + (None, None)
        show_hist = False
    else:
        show_hist = st.toggle("show histogram")
    readable_tokens = st.toggle("more readable tokens")
    amplifier = st.slider("amplify color", min_value=1.0, max_value=100.0, value=1.0, step=0.1)
    show_explanation = st.toggle("show explanation", value=True)

str_tokens, pos_idx, norm, counts, bin_edges = query_obj["query_info"]

if show_hist:
    fig = go.Figure(go.Bar(x=bin_edges[:-1], y=counts, width=np.diff(np.array(bin_edges))))
    fig.update_layout(title=f"Histogram of {'sim' if cosine else 'dist'} given query act {sel_query_id}",
                      width=600, height=300)
    st.plotly_chart(fig, use_container_width=False)

st.markdown("##### Query Activation")

seq_len = len(str_tokens)
if prev_ctx != "inf":
    span_s = max(0, pos_idx - int(prev_ctx))
else:
    span_s = 0
if futr_ctx != "inf":
    span_e = min(seq_len, pos_idx + 1 + int(futr_ctx))
else:
    span_e = seq_len
str_tokens = str_tokens[span_s: span_e]
if readable_tokens:
    str_tokens = more_readable_gpt2_tokens(str_tokens, gpt2_byte_decoder)
else:
    str_tokens = list(map(lambda x: x.replace("Ġ", " "), str_tokens))  # if x != "<|endoftext|>" else "[bos]"
str_tokens = process_special_tokens(str_tokens)

norm = f"norm: {norm:.1f} " if show_norm else ""

if cosine:
    shown_value = f"<span><i> sim: {1.0:.3f} ; {norm}pos: {pos_idx} ; </i></span>"
else:
    shown_value = f"<span><i> dist: {0.0:.3f} ; {norm}pos: {pos_idx} ; </i></span>"

best_sim = 0
best_sim_i = None
for i, item in enumerate(query_obj["preimage"]):
    if item[2] > best_sim:
        best_sim = item[2]
        best_sim_i = i
attr = list(map(lambda x: abs(x), query_obj["preimage"][best_sim_i][-1]))
normalizer = sum(attr) / amplifier
attr = attr[span_s: span_e]

spans = []
for i, (t, a) in enumerate(zip(str_tokens, attr)):
    spans.append(span_maker_attr(t, a, normalizer, i==pos_idx-span_s))
row = f'<div>' + shown_value + "".join(spans) + '</div>'
st.markdown(row, unsafe_allow_html=True)
st.text("")

samples = [item for i, item in enumerate(query_obj["preimage"]) if i != best_sim_i] # except query itself
random.shuffle(samples)
samples = sorted(samples[:num_show], key=lambda x: x[2], reverse=cosine)

st.markdown(f"##### Subspace {sel_subspace.split('-')[0]} Preimage")
not_below = True
for _, _, sim, str_tokens, pos_idx, norm, attr in samples:

    assert cosine # now only work for cosine
    if sim < 0.8 and not_below:
        not_below = False
        st.warning("Warning: similarity is too low")

    seq_len = len(str_tokens)
    if prev_ctx != "inf":
        span_s = max(0, pos_idx - int(prev_ctx))
    else:
        span_s = 0
    if futr_ctx != "inf":
        span_e = min(seq_len, pos_idx + 1 + int(futr_ctx))
    else:
        span_e = seq_len
    str_tokens = str_tokens[span_s: span_e]
    if readable_tokens:
        str_tokens = more_readable_gpt2_tokens(str_tokens, gpt2_byte_decoder)
    else:
        str_tokens = list(map(lambda x: x.replace("Ġ", " "), str_tokens))  # if x != "<|endoftext|>" else "[bos]"
    str_tokens = process_special_tokens(str_tokens)

    attr = list(map(lambda x: abs(x), attr))
    normalizer = sum(attr) / amplifier

    attr = attr[span_s: span_e]

    norm = f"norm: {norm:.1f} " if show_norm else ""
    
    shown_value = f"<span><i> {'sim' if cosine else 'dist'}: {sim:.3f} ; {norm}pos: {pos_idx} ; </i></span>"

    spans = []
    for i, (t, a) in enumerate(zip(str_tokens, attr)):
        spans.append(span_maker_attr(t, a, normalizer, i==pos_idx-span_s))

    row = f'<div style="margin-bottom: 8px;">' + shown_value + "".join(spans) + '</div>'
    st.markdown(row, unsafe_allow_html=True)


if show_explanation:
    st.write("#####")
    st.caption("""Token in bold and underscore is the token position where the activation is taken from. Blue color represents attribution score, you can hover each token to see un-normalized score.
               Unlike the other page, you can't choose arbitrary index for query activation, as the attribution scores are pre-computed and saved only for some randomly sampled query indices""")