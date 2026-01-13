from pathlib import Path

import streamlit as st
from huggingface_hub import snapshot_download


def render_import_tab() -> None:
    """Dataset import tab using Hugging Face snapshot_download."""

    st.subheader("Download Hugging Face Dataset")
    st.markdown(
        "Enter the repository ID (e.g., `TargetU/RcCArDataset`) and choose a local folder name."
    )

    col1, col2 = st.columns(2)
    with col1:
        repo_id = st.text_input("Repo ID", value="TargetU/RcCArDataset")
    with col2:
        local_dir = st.text_input("Local directory name (recommended: save under DataSet)", value="./DataSet/RcCArDataset")

    use_symlinks = st.checkbox("Use symbolic links (if available)", value=False)

    if st.button("Download Dataset", type="primary"):
        if not repo_id.strip():
            st.error("Please enter a valid repository ID.")
            return

        target_path = Path(local_dir).expanduser().resolve()
        st.info(f"📥 Download starting: {repo_id} → {target_path}")

        try:
            snapshot_download(
                repo_id=repo_id.strip(),
                repo_type="dataset",
                local_dir=str(target_path),
                local_dir_use_symlinks=use_symlinks,
            )
            st.success(f"✅ Download complete: {target_path}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Download error: {exc}")
            st.stop()
