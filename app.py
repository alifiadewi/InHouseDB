"""
File naming convention (one file per unit):
    <Group>_<DD-MMM-YYYY>_<SampleNumber>[_<Annotation>].xlsx

Run
    streamlit run app.py
"""
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

VALUE_COLUMNS = ["AnglePTP", "RearHubPTP"]
X_TICK_STEP   = 5
X_MINOR_STEP  = 1
SUPPORTED_EXT = {".xlsx", ".xls", ".xlsm", ".csv"}
MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

# ========================================================== Data Processing
def parse_filename(stem: str):
    m = re.match(
        r"^(?P<group>.+?)_(?P<date>\d{1,2}-[A-Za-z]{3}-\d{4})_"
        r"(?P<sample>[^_]+)(?:_(?P<annotation>.+))?$", stem)
    if not m: return None
    try:
        d, mon, y = m.group("date").split("-")
        month_num = MONTHS[mon.lower()[:3]]
        dt = datetime(int(y), month_num, int(d))
    except Exception:
        return None
    return dict(group=m.group("group"), date=dt, sample=m.group("sample"), 
                annotation=m.group("annotation") or "")

def clean_str(s):
    return str(s).lower().replace(" ", "").replace("_", "")

def find_frequency_column(cols):
    for c in cols:
        lc = clean_str(c)
        if "frequency" in lc and not any(x in lc for x in ("start", "end", "interval", "duty")):
            return c
    return None

def find_value_column(cols, target):
    t = clean_str(target)
    for c in cols:
        if t in clean_str(c):
            return c
    return None

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def read_measurement(source):
    try:
        suffix = Path(source if isinstance(source, (str, Path)) else source.name).suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(source, sep=None, engine='python')
        else:
            df = pd.read_excel(source)
    except Exception as e:
        return {"error": f"Cannot read file: {e}"}

    cols = list(df.columns)
    fcol = find_frequency_column(cols)
    if fcol is None:
        return {"error": f"No frequency column found. Available: {cols}"}

    freq = to_num(df[fcol])
    curves = {}
    for name in VALUE_COLUMNS:
        vcol = find_value_column(cols, name)
        if vcol is None:
            return {"error": f"Column '{name}' not found. Available: {cols}"}
        val = to_num(df[vcol])
        mask = freq.notna() & val.notna()
        f = freq[mask].to_numpy(dtype=float)
        v = val[mask].to_numpy(dtype=float)
        order = np.argsort(f)
        curves[name] = (f[order], v[order])

    return {"curves": curves}

def load_units(items):
    files = [f for f in items if not f.name.startswith("~$")]

    units, skipped = [], []
    for f in sorted(files, key=lambda p: Path(p.name).name):
        stem = Path(f.name).stem
        meta = parse_filename(stem)
        if meta is None:
            skipped.append(f"{f.name}: Bad filename format")
            continue
        data = read_measurement(f)
        if "error" in data:
            skipped.append(f"{f.name}: {data['error']}")
            continue
        meta["file"] = f.name
        meta["curves"] = data["curves"]
        units.append(meta)

    # --- NATURAL SORTING FIX (1, 2, 3... 10, 11 instead of 1, 10, 11, 2) ---
    def natural_sort_key(u):
        try:
            s_num = int(u['sample'])
        except ValueError:
            s_num = float('inf')
        return (u['date'], u['group'], s_num, u['sample'])
    
    units = sorted(units, key=natural_sort_key)

    return units, f"Loaded {len(units)} units.", skipped

def make_colors(n):
    if n == 0: return []
    scale = pcolors.sequential.Turbo
    ts = [0.5] if n == 1 else np.linspace(0.0, 1.0, n).tolist()
    return pcolors.sample_colorscale(scale, ts)

# ========================================================== Plotting
def build_figure(contexts, legend_on):
    n_contexts = len(contexts)
    
    # Auto-layout logic
    if n_contexts == 1:
        rows, cols = 2, 1
        v_space = 0.18 
        height = 820
    else:
        rows, cols = n_contexts, 2
        v_space = 0.15
        height = 420 * n_contexts
        
    titles = []
    for c in contexts:
        titles.extend([f"AnglePTP – {c['title']}", f"RearHubPTP – {c['title']}"])

    fig = make_subplots(
        rows=rows, cols=cols, 
        subplot_titles=titles,
        vertical_spacing=v_space, 
        horizontal_spacing=0.08
    )

    layout_kwargs = {}
    axis_idx = 1
    
    for ctx_idx, ctx in enumerate(contexts):
        # Define a unique legend name for this specific combination
        legend_name = "legend" if ctx_idx == 0 else f"legend{ctx_idx+1}"
        
        # Reset colors for EACH context
        ctx_colors = make_colors(len(ctx['indices']))
        
        # Calculate legend positioning (stack them vertically on the right)
        y_pos = 1.0 if n_contexts == 1 else 1.0 - (ctx_idx / n_contexts)
        layout_kwargs[legend_name] = dict(
            x=1.02, 
            y=y_pos, 
            xanchor="left", 
            yanchor="top",
            title=dict(text=ctx['title'], font=dict(size=12)),
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        )

        for v_i, name in enumerate(VALUE_COLUMNS):
            r = ctx_idx * 2 + v_i + 1 if n_contexts == 1 else ctx_idx + 1
            c = 1 if n_contexts == 1 else v_i + 1
            
            # Only show legend on the Left/Angle graph
            show_this_legend = legend_on and (v_i == 0)
            
            lo, hi = np.inf, -np.inf
            for color_idx, i in enumerate(ctx['indices']):
                u = ctx['units'][i]
                if name not in u["curves"]: continue
                f, v = u["curves"][name]
                label = u["sample"] + (f" ({u['annotation']})" if u["annotation"] else "")
                
                fig.add_trace(
                    go.Scatter(
                        x=f, y=v, mode="lines", 
                        line=dict(color=ctx_colors[color_idx], width=1.2),
                        name=label, 
                        showlegend=show_this_legend,
                        legend=legend_name, # Assign to specific legend
                        hovertemplate=f"<b>{label}</b><br>Freq: %{{x:.2f}} Hz<br>{name}: %{{y:.3f}}<extra></extra>"
                    ),
                    row=r, col=c
                )
                lo, hi = min(lo, float(f.min())), max(hi, float(f.max()))

            xname = f"xaxis{axis_idx}" if axis_idx > 1 else "xaxis"
            yname = f"yaxis{axis_idx}" if axis_idx > 1 else "yaxis"
            axis_idx += 1
            
            xcfg = dict(
                title="Frequency (Hz)", tickangle=90, 
                showgrid=True, gridcolor='#e5e5e5', zeroline=False
            )
            if np.isfinite(lo) and hi > lo:
                xcfg["range"] = [lo, hi]
                xcfg["dtick"] = X_TICK_STEP
                xcfg["minor"] = dict(ticks="outside", dtick=X_MINOR_STEP, showgrid=True, gridcolor='#f0f0f0')
            
            ycfg = dict(title=name, showgrid=True, gridcolor='#e5e5e5', zeroline=False)
            
            fig.update_layout(**{xname: xcfg})
            fig.update_layout(**{yname: ycfg})

    fig.update_layout(
        height=height, 
        margin=dict(l=60, r=220 if legend_on else 30, t=40, b=60),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=legend_on,
        **layout_kwargs
    )
    return fig

# ========================================================== UI & Layout
st.set_page_config(page_title="PTP Measurement Dashboard", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 2.5rem !important; padding-bottom: 1rem !important; max-width: 98% !important; }
    .stButton>button { width: 100%; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] ul {
        display: none !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        min-height: auto !important;
    }
    [data-testid="stFileUploadDropzone"] > div:first-child {
        display: none !important;
    }
    [data-testid="stFileUploadDropzone"] button {
        width: 100% !important;
        min-height: 38px !important;
        border: 1px solid rgba(49, 51, 63, 0.2) !important;
        border-radius: 0.5rem !important;
        background-color: transparent !important;
        position: relative;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploadDropzone"] button span {
        display: none !important;
    }
    [data-testid="stFileUploadDropzone"] button::after {
        content: "Select data folder..." !important;
        color: #31333F !important;
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        font-size: 14px !important;
        font-weight: 400 !important;
    }
    [data-testid="stFileUploadDropzone"] button:hover {
        border-color: #ff4b4b !important;
    }
    [data-testid="stFileUploadDropzone"] button:hover::after {
        color: #ff4b4b !important;
    }
</style>
""", unsafe_allow_html=True)

if "units" not in st.session_state: st.session_state.units = []
if "show" not in st.session_state: st.session_state.show = {}
if "errors" not in st.session_state: st.session_state.errors = []
if "folder_name" not in st.session_state: st.session_state.folder_name = ""

col_left, col_right = st.columns([1.2, 4], gap="medium")

with col_left:
    uploaded_files = st.file_uploader(
        "Upload folder", 
        type=["xlsx", "xls", "xlsm", "csv"], 
        accept_multiple_files='directory', 
        label_visibility="collapsed"
    )

    if uploaded_files:
        units, msg, errs = load_units(uploaded_files)
        st.session_state.units = units
        st.session_state.show = {i: True for i in range(len(units))}
        st.session_state.errors = errs
        st.session_state.folder_name = msg

    if st.session_state.folder_name:
        st.caption(f"{st.session_state.folder_name}")
    else:
        st.caption("No folder selected")
        
    st.write("")

    if st.session_state.units:
        dates = sorted({u["date"] for u in st.session_state.units})
        sel_dates = st.multiselect("Batch date", dates, default=dates)
        
        groups = sorted({u["group"] for u in st.session_state.units})
        sel_groups = st.multiselect("Group", groups, default=groups)

        if not sel_dates: sel_dates = dates
        if not sel_groups: sel_groups = groups

        contexts_dict = {}
        for i, u in enumerate(st.session_state.units):
            if u['date'] in sel_dates and u['group'] in sel_groups:
                ctx_key = (u['date'], u['group'])
                if ctx_key not in contexts_dict:
                    contexts_dict[ctx_key] = {
                        'indices': [], 
                        'title': f"{u['group']} ({u['date'].strftime('%d-%b-%Y')})"
                    }
                contexts_dict[ctx_key]['indices'].append(i)

        sorted_keys = sorted(contexts_dict.keys(), key=lambda x: (x[0], x[1]))
        filtered = [i for k in sorted_keys for i in contexts_dict[k]['indices']]
    else:
        sorted_keys = []
        contexts_dict = {}
        filtered = []

    st.markdown("**Samples**")
    if filtered:
        table_df = pd.DataFrame({
            "Show": [st.session_state.show.get(i, True) for i in filtered],
            "Sample": [st.session_state.units[i]["sample"] for i in filtered],
            "Annotation": [st.session_state.units[i]["annotation"] for i in filtered],
        })
        edited = st.data_editor(
            table_df,
            column_config={
                "Show": st.column_config.CheckboxColumn("Show", default=True),
                "Sample": st.column_config.TextColumn("Sample", disabled=True),
                "Annotation": st.column_config.TextColumn("Annotation", disabled=True),
            },
            hide_index=True, use_container_width=True, height=min(42 + 35 * len(table_df), 350),
            key=f"editor_{str(sel_dates)}_{str(sel_groups)}"
        )
        for i, show in zip(filtered, edited["Show"].tolist()):
            st.session_state.show[i] = bool(show)
    else:
        st.data_editor(
            pd.DataFrame(columns=["Show", "Sample", "Annotation"]),
            column_config={
                "Show": st.column_config.CheckboxColumn("Show", default=False),
                "Sample": st.column_config.TextColumn("Sample", disabled=True),
                "Annotation": st.column_config.TextColumn("Annotation", disabled=True),
            },
            hide_index=True, use_container_width=True, height=350, key="empty_table"
        )
        
    b1, b2, b3 = st.columns(3)
    if b1.button("All", use_container_width=True):
        if filtered:
            for i in filtered: st.session_state.show[i] = True
            st.rerun()
    if b2.button("None", use_container_width=True):
        if filtered:
            for i in filtered: st.session_state.show[i] = False
            st.rerun()
    if b3.button("Invert", use_container_width=True):
        if filtered:
            for i in filtered: st.session_state.show[i] = not st.session_state.show.get(i, True)
            st.rerun()
            
    legend_on = st.checkbox("Show legend", value=False)

with col_right:
    active_contexts = []
    for k in sorted_keys:
        ctx = contexts_dict[k]
        shown_in_ctx = [i for i in ctx['indices'] if st.session_state.show.get(i, True)]
        active_contexts.append({
            'title': ctx['title'],
            'indices': shown_in_ctx,
            'units': st.session_state.units
        })
    
    if active_contexts:
        fig = build_figure(active_contexts, legend_on)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data selected to plot.")

    if st.session_state.errors:
        with st.expander(f"⚠️ {len(st.session_state.errors)} files were skipped. Click to see why."):
            for err in st.session_state.errors:
                st.write(f"- {err}")
