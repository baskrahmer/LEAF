import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import Config
from src.data import get_ciqual_data


def plot_lang_label_frequencies(lang_label_frequencies):
    data = []
    for lang, label_freqs in lang_label_frequencies.items():
        for label, freq in label_freqs.items():
            data.append([lang, label, freq])
    df = pd.DataFrame(data, columns=['Language', 'Label', 'Frequency'])

    # Creating a pivot table for the heatmap
    pivot_table = df.pivot(index='Label', columns='Language', values='Frequency').fillna(0)

    # Dividing each cell by the total for its label and multiplying by 100 to get percentages
    pivot_table_percent = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

    # Summing the frequencies for each language and label
    language_totals = pivot_table_percent.sum(axis=0).sort_values(ascending=False)
    label_totals = pivot_table_percent.sum(axis=1).sort_values(ascending=False)

    # Reordering the pivot table based on the sorted totals
    pivot_table_sorted = pivot_table_percent.reindex(index=label_totals.index, columns=language_totals.index)

    # Plotting the sorted heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table_sorted, cmap="YlGnBu", cbar_kws={'label': 'Percentage'})
    plt.title('Percentage of Language-Label Combinations (Sorted)')
    plt.ylabel('Label')
    plt.xlabel('Language')
    plt.show()

    # Plotting for the 5 most frequent and 5 least frequent classes
    for label_subset, title_suffix in [(label_totals.head(5).index, '5 Most Frequent Classes'),
                                       (label_totals.tail(5).index, '5 Least Frequent Classes')]:
        # Filtering the pivot table for the selected labels
        pivot_subset = pivot_table_sorted.loc[label_subset]

        # Plotting for all languages
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_subset, cmap="YlGnBu", cbar_kws={'label': 'Percentage'})
        plt.title(f'Percentage of Language-Label Combinations ({title_suffix})')
        plt.ylabel('Label')
        plt.xlabel('Language')
        plt.show()

        # Plotting for the 5 most frequent languages
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_subset[language_totals.head(5).index], cmap="YlGnBu", cbar_kws={'label': 'Percentage'})
        plt.title(f'Percentage of Language-Label Combinations ({title_suffix}) - Top 5 Languages')
        plt.ylabel('Label')
        plt.xlabel('Language')
        plt.show()


def plot_ciqual_distribution(c: Config, save_path: str):
    footprint_scores = get_ciqual_data(c)["Score unique EF"]
    num_bins = 10
    plt.hist(footprint_scores, bins=num_bins, edgecolor='black')

    # Adding labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Co2e distribution')
    bins = np.linspace(min(footprint_scores), max(footprint_scores), num_bins + 1)
    ticks = [(bins[i] + bins[i + 1]) / 2 for i in range(num_bins)]
    formatter = ticker.FormatStrFormatter('%.2f')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xticks(ticks)
    plt.yscale("log")

    # Saving the plot
    plt.savefig(os.path.join(save_path, 'ciqual_distribution.png'))
    plt.close()


def plot_lang_label_frequencies(lang_label_frequencies, save_path):
    data = []
    for lang, label_freqs in lang_label_frequencies.items():
        for label, freq in label_freqs.items():
            data.append([lang, label, freq])
    df = pd.DataFrame(data, columns=['Language', 'Label', 'Frequency'])

    # Creating a pivot table for the heatmap
    pivot_table = df.pivot(index='Label', columns='Language', values='Frequency').fillna(0)

    # Dividing each cell by the total for its label and multiplying by 100 to get percentages
    pivot_table_percent = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

    # Summing the frequencies for each language and label
    language_totals = pivot_table_percent.sum(axis=0).sort_values(ascending=False)
    label_totals = pivot_table_percent.sum(axis=1).sort_values(ascending=False)

    # Reordering the pivot table based on the sorted totals
    pivot_table_sorted = pivot_table_percent.reindex(index=label_totals.index, columns=language_totals.index)

    # Plotting the sorted heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table_sorted, cmap="YlGnBu", cbar_kws={'label': 'Percentage'})
    plt.title('Percentage of Language-Label Combinations (Sorted)')
    plt.ylabel('Label')
    plt.xlabel('Language')
    plt.savefig(os.path.join(save_path, 'lang_label_frequencies_sorted.png'))
    plt.close()

    # Plotting for the 5 most frequent and 5 least frequent classes
    for label_subset, title_suffix in [(label_totals.head(5).index, '5 Most Frequent Classes'),
                                       (label_totals.tail(5).index, '5 Least Frequent Classes')]:
        # Filtering the pivot table for the selected labels
        pivot_subset = pivot_table_sorted.loc[label_subset]

        # Plotting for all languages
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_subset, cmap="YlGnBu", cbar_kws={'label': 'Percentage'})
        plt.title(f'Percentage of Language-Label Combinations ({title_suffix})')
        plt.ylabel('Label')
        plt.xlabel('Language')
        plt.savefig(os.path.join(save_path, f'lang_label_frequencies_{title_suffix.replace(" ", "_").lower()}.png'))
        plt.close()

        # Plotting for the 5 most frequent languages
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_subset[language_totals.head(5).index], cmap="YlGnBu", cbar_kws={'label': 'Percentage'})
        plt.title(f'Percentage of Language-Label Combinations ({title_suffix}) - Top 5 Languages')
        plt.ylabel('Label')
        plt.xlabel('Language')
        plt.savefig(os.path.join(save_path,
                                 f'lang_label_frequencies_{title_suffix.replace(" ", "_").lower()}_top_5_languages.png'))
        plt.close()


def save_dict_to_json(data, save_path, filename):
    with open(os.path.join(save_path, filename), 'w') as f:
        json.dump(data, f)


def make_data_analysis_report(c: Config, lang_frequencies: dict, label_frequencies: dict, lang_label_frequencies: dict,
                              output_path: str, mlm: bool):
    # Ensure the save directory exists
    os.makedirs(output_path, exist_ok=True)

    # Saving dictionaries to JSON files
    save_dict_to_json(lang_frequencies, output_path, 'lang_frequencies.json')
    if not mlm:
        save_dict_to_json(label_frequencies, output_path, 'label_frequencies.json')
        save_dict_to_json(lang_label_frequencies, output_path, 'lang_label_frequencies.json')

        # Plotting and saving plots
        plot_ciqual_distribution(c, output_path)
        plot_lang_label_frequencies(lang_label_frequencies, output_path)
