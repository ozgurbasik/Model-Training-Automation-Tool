import streamlit as st

from tabs.dataset_transformation_tab import render_dataset_transformation_tab
from tabs.experiments_tab import render_experiments_tab
from tabs.import_tab import render_import_tab
from tabs.train_tab import render_train_tab
from tabs.validation_tab import render_dataset_validation_tab
from tabs.split_tab import render_dataset_split_tab
from tabs.test_tab import render_test_tab


def main():
    st.title("Autonomous Vehicle Training Dashboard")

    task_display = st.sidebar.selectbox("Task type", ["Object Detection", "Segmentation"])
    task_type = "detection" if task_display == "Object Detection" else "segmentation"

    tab_import, tab_transform, tab_validate, tab_split, tab_train, tab_experiments, tab_test = st.tabs(
        ["Dataset Import", "Dataset Transformation", "Dataset Validation", "Dataset Split", "Train", "Experiments", "Test"]
    )

    with tab_import:
        render_import_tab()

    with tab_transform:
        render_dataset_transformation_tab()

    with tab_validate:
        render_dataset_validation_tab()

    with tab_split:
        render_dataset_split_tab()

    with tab_train:
        render_train_tab(task_type, task_display)

    with tab_experiments:
        st.subheader("Experiments")
        render_experiments_tab(task_type)
    
    with tab_test:
        render_test_tab(task_type)


if __name__ == "__main__":
    main()


