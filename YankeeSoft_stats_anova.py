# run below on shell: streamlit run App_MACOVA.py

import streamlit as st
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.multivariate.manova import MANOVA
import numpy as np
from scipy.stats import ttest_ind, f_oneway

def highlight_significant(val):
    return 'background-color: lightcoral' if val < 0.05 else ''

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(((nx - 1)*x.std(ddof=1)**2 + (ny - 1)*y.std(ddof=1)**2) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled_sd if pooled_sd != 0 else np.nan

st.title("Flexible MANOVA / ANCOVA / T-test Analysis Tool v1.0")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # df = pd.read_csv(uploaded_file)
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')  # or encoding='latin1'

    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')

    with st.sidebar:
        st.markdown("### Column Selection")

        if st.checkbox("Show head of CSV"):
            st.write(df.head())

        first_five = df.columns[:10].tolist()
        category_matches = [col for col in first_five if col.lower() == "category"]
        if category_matches:
            default_idx = first_five.index(category_matches[0])
        else:
            try:
                default_idx = next(i for i, col in enumerate(first_five) if df[col].dtype in ['object', 'category'])
            except StopIteration:
                st.warning("No categorical columns found in the first 5 columns.")
                st.stop()
        category_col = st.selectbox("Select Grouping Column", options=first_five, index=default_idx)

        # original_groups = df[category_col].dropna().unique().tolist()
        # selected_merge_groups = st.multiselect("Merge Categories (hold ctrl/cmd to multi-select)", options=original_groups)
        # merged_label = st.text_input("Label for Merged Group", value="Merged")
        #
        # if selected_merge_groups:
        #     df[category_col] = df[category_col].replace({g: merged_label for g in selected_merge_groups})

        dependent_variable_cols = st.multiselect("Select Dependent Variables", options=df.columns, default=df.columns[7:])
        confound_cols = st.multiselect("Select Confounding Variables", options=df.columns, default=df.columns[2:6])
        run_manova = st.checkbox("Run MANOVA (Multivariate Test)", value=False)

        st.markdown("### Filter Subjects")
        filter_df = df.copy()

        if "Age" in df.columns:
            min_age, max_age = int(df['Age'].min() - 1), int(df['Age'].max() + 1)
            age_range = st.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
            filter_df = filter_df[(filter_df['Age'] >= age_range[0]) & (filter_df['Age'] <= age_range[1])]

        if "Sex" in df.columns:
            sex_options = df["Sex"].dropna().unique().tolist()
            selected_sex = st.multiselect("Select Sex", sex_options, default=sex_options)
            filter_df = filter_df[filter_df["Sex"].isin(selected_sex)]

        if "Race" in df.columns:
            race_options = df["Race"].dropna().unique().tolist()
            selected_race = st.multiselect("Select Race", race_options, default=race_options)
            filter_df = filter_df[filter_df["Race"].isin(selected_race)]

        if "Scanner" in df.columns:
            scanner_options = df["Scanner"].dropna().unique().tolist()
            selected_scanner = st.multiselect("Select Scanner", scanner_options, default=scanner_options)
            filter_df = filter_df[filter_df["Scanner"].isin(selected_scanner)]

        filter_column = st.selectbox("Optional: Select a Column to Filter", options=df.columns)
        if filter_column:
            if np.issubdtype(df[filter_column].dtype, np.number):
                min_val, max_val = float(df[filter_column].min()), float(df[filter_column].max())
                range_vals = st.slider(f"Select range for {filter_column}", min_val, max_val, (min_val, max_val))
                filter_df = filter_df[(filter_df[filter_column] >= range_vals[0]) & (filter_df[filter_column] <= range_vals[1])]
            else:
                unique_values = df[filter_column].dropna().unique().tolist()
                selected_values = st.multiselect(f"Select values for {filter_column}", options=unique_values, default=unique_values)
                filter_df = filter_df[filter_df[filter_column].isin(selected_values)]

        st.markdown(f"✅ {len(filter_df)} subjects after filtering")

    filter_df[category_col] = filter_df[category_col].astype(str).str.strip().str.capitalize().astype('category')

    st.markdown("### Sample Count per Category")
    category_counts_df = filter_df[category_col].value_counts(dropna=False).reset_index()
    category_counts_df.columns = [category_col, "Count"]
    st.dataframe(category_counts_df)

    if dependent_variable_cols and category_col and len(filter_df) > 2:

        if run_manova:
            st.subheader("MANOVA Result")
            manova_formula = ' + '.join(dependent_variable_cols) + ' ~ ' + category_col
            if confound_cols:
                manova_formula += ' + ' + ' + '.join(confound_cols)
            st.markdown(f"**MANOVA Formula:** `{manova_formula}`")
            try:
                manova = MANOVA.from_formula(manova_formula, data=filter_df)
                st.code(str(manova.mv_test()), language="text")
            except Exception as e:
                st.error(f"MANOVA failed: {e}")

        st.subheader("Per-Variable Analysis (ANCOVA or T-test)")
        raw_pvals, test_stats, test_type, effect_sizes = [], [], [], []

        for idx, dependent_variable in enumerate(dependent_variable_cols):
            if confound_cols:
                ancova_formula = f"{dependent_variable} ~ {category_col} + " + ' + '.join(confound_cols)
                model = ols(ancova_formula, data=filter_df).fit()
                table = sm.stats.anova_lm(model, typ=2)
                if category_col in table.index:
                    stat = table.loc[category_col, "F"]
                    pval = table.loc[category_col, "PR(>F)"]
                    ss_effect = table.loc[category_col, 'sum_sq']
                    ss_error = table.loc['Residual', 'sum_sq']
                    eta_sq = ss_effect / (ss_effect + ss_error)
                else:
                    stat, pval, eta_sq = np.nan, np.nan, np.nan
                test_stats.append(stat)
                raw_pvals.append(pval)
                effect_sizes.append(eta_sq)
                test_type.append("ANCOVA")
                if idx == 0:
                    st.markdown(f"**ANCOVA Formula for `{dependent_variable}`:** `{ancova_formula}`")
            else:
                groups = filter_df[category_col].dropna().unique()
                group_data = [filter_df[filter_df[category_col] == g][dependent_variable].dropna() for g in groups]
                if len(groups) == 2:
                    stat, pval = ttest_ind(group_data[0], group_data[1], equal_var=False)
                    d = cohens_d(group_data[0], group_data[1])
                    test_stats.append(stat)
                    raw_pvals.append(pval)
                    effect_sizes.append(d)
                    test_type.append("T-test")
                elif len(groups) > 2:
                    stat, pval = f_oneway(*group_data)
                    test_stats.append(stat)
                    raw_pvals.append(pval)
                    effect_sizes.append(np.nan)
                    test_type.append("ANOVA")
                else:
                    test_stats.append(None)
                    raw_pvals.append(None)
                    effect_sizes.append(None)
                    test_type.append("Invalid")

        _, fdr_pvals, _, _ = multipletests(raw_pvals, method='fdr_bh')

        result_df = pd.DataFrame({
            "Dependent Variable": dependent_variable_cols,
            "Test Type": test_type,
            "Statistic": test_stats,
            "Raw p-value": raw_pvals,
            "FDR corrected p-value": fdr_pvals,
            "Effect Size": effect_sizes
        })

        st.dataframe(
            result_df.style
            .format({
                "Statistic": "{:.2f}",
                "Raw p-value": "{:.4f}",
                "FDR corrected p-value": "{:.4f}",
                "Effect Size": "{:.3f}"
            })
            .map(highlight_significant, subset=["FDR corrected p-value"])
        )

        st.subheader("Mean Value Plot")
        selected_net = st.selectbox("Select a Variable to visualize", dependent_variable_cols)
        st.bar_chart(filter_df.groupby(category_col)[selected_net].mean())


        ## Functions to be added
        ## 1. The Wilcoxon test is a non-parametric statistical test used to compare two groups when the assumptions of
        # normal distribution are not met.

        ## 2. Mendelian Randomization (MR) is a statistical method used in epidemiology and genetics to infer causal relationships
        # between a modifiable exposure (e.g., blood pressure) and an outcome (e.g., cognitive decline), using genetic variants
        # as natural instruments.


        ## 3. Partial correlation analysis.


