import streamlit as st
import pandas as pd
import os
import json
from PIL import Image

st.set_page_config(page_title="ETL Data Dashboard", layout="wide")

DATA_ROOT = "/data"

def list_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def show_image(img_path, caption=None):
    try:
        img = Image.open(img_path)
        st.image(img, caption=caption or os.path.basename(img_path), use_column_width=True)
    except Exception as e:
        st.error(f"Could not load image: {img_path}")

st.title("Project3 ETL Data Dashboard")

# --- Dataset Selection ---
dataset = st.sidebar.selectbox(
    "Select Dataset",
    ["AiVsHuman", "Flickr30k"]
)

if dataset == "AiVsHuman":
    base_dir = os.path.join(DATA_ROOT, "AiVsHuman")
    st.header("AiVsHuman Dataset")
    # List available splits
    splits = []
    for fname in os.listdir(base_dir):
        if fname.endswith(".csv"):
            splits.append(fname.replace(".csv", ""))
    split = st.sidebar.selectbox("Select Split", splits)
    csv_path = os.path.join(base_dir, f"{split}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.write(f"**Rows:** {len(df)}")
        st.write(f"**Columns:** {list(df.columns)}")
        st.write("**Missing values:**")
        st.write(df.isnull().sum())
        st.write("**Label distribution:**")
        if "label" in df.columns:
            st.bar_chart(df["label"].value_counts())
        # Show random sample images
        st.subheader("Sample Images")
        n_samples = st.slider("Number of samples", 1, 10, 3, key="aivshuman_samples")
        if "file_name" in df.columns:
            samples = df.sample(n=n_samples)
            img_dir = os.path.join(base_dir, "Images")
            for _, row in samples.iterrows():
                img_path = os.path.join(img_dir, row["file_name"])
                label = str(row["label"]) if "label" in row else ""
                show_image(img_path, caption=f"Label: {label}")
    else:
        st.warning(f"No CSV found at {csv_path}")

elif dataset == "Flickr30k":
    base_dir = os.path.join(DATA_ROOT, "Flickr30k")
    st.header("Flickr30k Dataset")
    # List available splits
    splits = []
    for fname in os.listdir(base_dir):
        if fname.endswith(".json") and "gt" not in fname:
            splits.append(fname.replace(".json", ""))
    split = st.sidebar.selectbox("Select Split", splits)
    json_path = os.path.join(base_dir, f"{split}.json")
    img_dir = os.path.join(base_dir, "flickr30k-images")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        st.write(f"**Total records:** {len(data)}")
        # Show sample images and captions
        st.subheader("Sample Images & Captions")
        n_samples = st.slider("Number of samples", 1, 10, 3, key="flickr_samples")
        samples = data[:n_samples] if len(data) >= n_samples else data
        for item in samples:
            img_path = os.path.join(img_dir, item["image"])
            captions = item["caption"]
            show_image(img_path, caption="; ".join(captions))
    else:
        st.warning(f"No JSON found at {json_path}")

st.markdown("---")
st.caption("Data and splits are loaded directly from ETL pipeline outputs.")
