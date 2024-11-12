from collections import Counter
from scipy.spatial.distance import jaccard
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from openTSNE import TSNE
from pathlib import Path

import pandas as pd
import numpy as np
import math
import umap
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import colorsys

def read_in_file(file_path):
    sequences = []
    all_labels = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()     

    # Find indices where sequence labels start ('>')
    label_indices = [i for i, line in enumerate(lines) if line.strip().startswith('>')]

    if not label_indices:
        raise ValueError("Sequence label line not found in the Clustal Omega output.")

    #append each label to all_labels
    for i in range(len(label_indices)):
        whole_label = lines[label_indices[i]]
        all_labels.append(whole_label)
    
    #append each sequence to sequences
    for i in range(len(label_indices)):
        start_index = label_indices[i]
        end_index = label_indices[i + 1] if i < len(label_indices) - 1 else None
        sequence = ''.join(line.strip() for line in lines[start_index + 1:end_index])
        sequences.append(sequence)
    
    print(f"{len(sequences)} sequences read from file")

    return sequences, all_labels


def find_limits(aligned_sequences, labels):
    
    #List to store indices of the first and last non-dash character in each sequence
    indices_of_first_non_dash = []
    indices_of_last_non_dash = []
    
    for sequence in aligned_sequences:
    
        # Find the index of the first non-dash character
        index_of_first_non_dash = next((i for i, char in enumerate(sequence) if char != '-'), None)
        if index_of_first_non_dash is not None:
            indices_of_first_non_dash.append(index_of_first_non_dash)
        else:
            indices_of_first_non_dash.append(None)
    
        # Find the index of the last non-dash character before end dashes
        reversed_sequence = sequence[::-1]
        index_of_last_non_dash = next((i for i, char in enumerate(reversed_sequence) if char != '-'), None)
        if index_of_last_non_dash is not None:
            index_of_last_non_dash = len(sequence) - index_of_last_non_dash - 1
            indices_of_last_non_dash.append(index_of_last_non_dash)
        else:
            indices_of_last_non_dash.append(None)
        
    #count most common results of where the nucleotides start and end
    index_counts = Counter(indices_of_first_non_dash)
    last_index_counts = Counter(indices_of_last_non_dash)
    
    #want second (index 1) most common start and end position to allow for more overlap
    most_common_index, occurrences = index_counts.most_common(2)[1]
    most_common_last_non_dash_index, last_occurrences = last_index_counts.most_common(2)[1]

    print(f"Second most common nucleotide start index: {most_common_index} (Occurrences: {occurrences})")
    print(f"Second most common nucleotide last index: {most_common_last_non_dash_index} (Occurrences: {last_occurrences})")
    
    #shorten sequences with new limits
    all_sequences = []
    for i in aligned_sequences:
        temp = i[most_common_index:most_common_last_non_dash_index+1] #include last nucleotide before -
        all_sequences.append(temp)

    #Lists to store label information
    states = []
    dates = []

    # Loop through each label and extract state name and date
    for label in labels:
        
        #Normal label example: >OQ667514 A/New York/52/2022 2022/12/13 4 (HA)
        
        #Find the indices of the first two forward slashes (where state in located in label)
        first_slash_index = label.find('/')
        second_slash_index = label.find('/', first_slash_index + 1)
    
        #Find the index of the space after the date
        space_before_date_index = label.find(' ', second_slash_index + 1)
        last_slash = label.rfind('/')
        end_of_date = label.find(' ', last_slash)

        #Append states and dates to list
        if first_slash_index != -1 and second_slash_index != -1 and space_before_date_index != -1 and end_of_date != -1:

            #Extract the words between the first two forward slashes (state)
            words_between_slashes = label[first_slash_index + 1:second_slash_index]
        
            # Error: >MW855331 A/Human/New York City/PV08753/2020 2020/03/09 4 (HA)
            if words_between_slashes == "Human" or words_between_slashes == "human":
                states.append("New York")
                third_slash_index = label.find('/', second_slash_index + 1)
                space_before_date_index = label.find(' ', third_slash_index + 1)
            elif words_between_slashes == "environment":
            #A/environment/Indiana/16TOSU3896/2016 2016/08/01 4 (HA)
                states.append("Indiana")
                third_slash_index = label.find('/', second_slash_index + 1)
                space_before_date_index = label.find(' ', third_slash_index + 1)
            else:
                states.append(words_between_slashes)
            
            # Extract the date
            date = label[space_before_date_index + 1:end_of_date]
        
            #Account for specific errors
            if date[0] == "S":
                next_space = label.find(' ', space_before_date_index + 1)
                date = label[next_space + 1: end_of_date]
            
            dates.append(date)
        
        #No state or no date was found
        else:
            states.append(None)
            dates.append(None)

    #Lists to store specific date information
    years = []
    months = []
    days = []

    #find year, month, and day from date and append to lists
    for i in dates:
        find_year = i.find('/')
        find_day = i.rfind('/')
        year = i[0:find_year]
        month = i[find_year+1:find_year+3]
        day = i[find_day:]
    
        years.append(year)
    
        #Account for errors
        if day == '/':
            days.append('None')
        else:
            days.append(day)
        if month == '/':
            months.append('None')
        else:
            months.append(month)
        
    df = pd.DataFrame({'years': years, 'months': months, 'days': days, 'state': states, 'seq': all_sequences}) 
    
    return df


def quarter_dates(df):

    dfDATES = df

    # Convert 'months' column to integers
    dfDATES['months'] = pd.to_numeric(dfDATES['months'], errors='coerce')
    
    # Create a new column for quarter of the year
    dfDATES['quarter'] = pd.to_datetime(dfDATES['months'], format='%m', errors='coerce').dt.quarter

    # Handle cases where 'months' is None
    dfDATES['quarter'] = dfDATES['quarter'].fillna(0).astype(int)

    # Display the result
    #print(dfDATES)
    
    #remove quarter == 0 cases
    dfDATES = dfDATES[dfDATES['quarter'] != 0]

    return dfDATES


def calc_diff(all_sequences, reference_sequence):

    sequences = all_sequences

    # Find the length among all sequences (they should all be the same after find_limits)
    length = len(reference_sequence)

    # Initialize a NumPy array filled with zeros
    differences_matrix = np.zeros((len(sequences), length), dtype=int)

    # Iterate through each sequence in sequences
    for i, sequence in enumerate(sequences):
        
        # Compare each letter in the sequence to the reference at that position
        for j in range(length):
            
            if (reference_sequence[j] == 'A' and sequence[j] == 'A') or \
               (reference_sequence[j] == 'C' and sequence[j] == 'C') or \
               (reference_sequence[j] == 'G' and sequence[j] == 'G') or \
               (reference_sequence[j] == 'T' and sequence[j] == 'T') or \
               (reference_sequence[j] == '-' and sequence[j] == '-'):
                differences_matrix[i, j] = 0  # no mutation
                
            elif reference_sequence[j] == 'A':
                if sequence[j] == 'C':
                    differences_matrix[i, j] = 1
                elif sequence[j] == 'G':
                    differences_matrix[i, j] = 2
                elif sequence[j] == 'T':
                    differences_matrix[i, j] = 3
                elif sequence[j] == '-':
                    differences_matrix[i, j] = 4
                    
            elif reference_sequence[j] == 'C':
                if sequence[j] == 'A':
                    differences_matrix[i, j] = 5
                elif sequence[j] == 'G':
                    differences_matrix[i, j] = 6
                elif sequence[j] == 'T':
                    differences_matrix[i, j] = 7
                elif sequence[j] == '-':
                    differences_matrix[i, j] = 8
                    
            elif reference_sequence[j] == 'G':
                if sequence[j] == 'C':
                    differences_matrix[i, j] = 9
                elif sequence[j] == 'A':
                    differences_matrix[i, j] = 10
                elif sequence[j] == 'T':
                    differences_matrix[i, j] = 11
                elif sequence[j] == '-':
                    differences_matrix[i, j] = 12
                    
            elif reference_sequence[j] == 'T':
                if sequence[j] == 'C':
                    differences_matrix[i, j] = 13
                elif sequence[j] == 'G':
                    differences_matrix[i, j] = 14
                elif sequence[j] == 'A':
                    differences_matrix[i, j] = 15
                elif sequence[j] == '-':
                    differences_matrix[i, j] = 16
                    
            elif reference_sequence[j] == '-':
                if sequence[j] == 'C':
                    differences_matrix[i, j] = 17
                elif sequence[j] == 'G':
                    differences_matrix[i, j] = 18
                elif sequence[j] == 'A':
                    differences_matrix[i, j] = 19
                elif sequence[j] == 'T':
                    differences_matrix[i, j] = 20

    print("Differences Matrix:")
    print(differences_matrix)
    
    return differences_matrix


def calc_jaccard(differences_matrix):
    
    r, c = differences_matrix.shape
    num_seqs = r

    #Initialize matrix
    distances = np.zeros((num_seqs, num_seqs))

    for i in range(num_seqs):
        for j in range(i + 1, num_seqs):
            distances[i][j] = jaccard(differences_matrix[i], differences_matrix[j])
            distances[j][i] = distances[i][j]  # Distance matrix is symmetric

    print("Jaccard Distances:")
    print(distances)
    
    return distances


def calc_PCA(distances):
    
    X = distances
    print(f"Shape before PCA: ", {X.shape})

    pca1 = PCA(n_components = 2) 

    X_newPCA = pca1.fit_transform(X)

    print(f"Shape before PCA: ", {X_newPCA.shape})
    
    return X_newPCA, pca1


def calc_UMAP(distances):
    
    X = distances
    print(f"Shape before UMAP: ", {X.shape})

    umap1 = umap.UMAP(n_components = 2, n_neighbors = 110, min_dist = 0.6, init = "pca")
    umap1.fit(X)
    X_newUMAP = umap1.transform(X)

    print(f"Shape after UMAP: ", {X_newUMAP.shape})
    
    return X_newUMAP, umap1


def calc_TSNE(distances):
    
    tsne_open = TSNE(
        perplexity=30,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=True,)

    X_newTSNE = tsne_open.fit(distances)
    
    return X_newTSNE


def elbow(X,Y):

    data = np.column_stack((X, Y))

    range_of_use = range(1,25)

    # Define range of clusters (K)
    k_values = [i for i in range_of_use]  # Adjust range of clusters

    # Calculate within-cluster sum of squares (WCSS) for each value of K
    wcss = []
    for k in range_of_use:
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(data) 
        wcss.append(kmeans.inertia_)  # Inertia is the WCSS

    # Plot the elbow curve
    plt.plot(k_values, np.log(wcss), marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(k_values)
    plt.show()
    
    
def find_centers_and_labels(num_clusters, X, Y):
    
    # Concatenate X and Y to create the feature matrix
    features = np.column_stack((X, Y))

    # Create KMeans object and fit it to the data
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(features)

    # Get the cluster centers and labels
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    print("Cluster centers:")
    print(centers)
    print("\nCluster labels:")
    print(labels)
    
    return centers, labels


def visualize_clusters(color_label, color_map, X, Y, centers, xlab, ylab, title, save_title, k_labels, group, legend):

    plt.figure(figsize=(8, 6))
    
    # Plot the data points
    plt.scatter(X, Y, c=color_label, alpha=0.3, label='Data Points', s = 17)

    # Plot the cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', label='Cluster Centers')
    
    #if legend = true from function call
    if legend:
        handles = []
        for cluster_label in np.unique(k_labels):
            if group == "bc":
                this_label = ('Pre Cluster {}'.format(cluster_label + 1))
            elif group == "ac":
                this_label=('Post Cluster {}'.format(cluster_label + 1))
            else: 
                this_label=('Cluster {}'.format(cluster_label + 1))
            # Create a color patch for each cluster
            patch = plt.Line2D([0], [0], marker='o', color='w', label=this_label, 
                               markerfacecolor=color_map[cluster_label], markersize=10)
            handles.append(patch)

        handles.append(plt.Line2D([], [], color='black', marker='x', linestyle='None', markersize=10, label='Cluster Centers'))

        # Add legend with custom handles
        legend = plt.legend(handles=handles, title='Clusters', loc='upper right', bbox_to_anchor=(1.35, 1), fontsize = 11, title_fontsize=13, prop={'family': 'serif'})

        #plt.setp(legend.get_title())
        legend.get_title().set_fontsize(13)
        legend.get_title().set_fontfamily('serif')

    #plt.title(title, fontsize = 15)
    plt.xlabel(xlab, fontsize = 14)
    plt.ylabel(ylab, fontsize = 14)
    
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.savefig(save_title, bbox_inches='tight')
    
    
def get_colors(num_clust, my_labels, group):
    
    if num_clust > 10:
        print("Too many clusters for color choices")
    
    else:
        this_set = plt.colormaps['tab20']
        

        if group == "ac" or group == "all":
            color_map = {i: this_set(j) for i, j in zip(range(num_clust), range(0, num_clust * 2, 2))}
            color_label = [color_map[label] for label in my_labels]
            
        elif group == "bc":
            color_map = {i: this_set(j) for i, j in zip(range(num_clust), range(1, (num_clust * 2)+1, 2))}
            color_label = [color_map[label] for label in my_labels]
    
    return color_label, color_map


#visualize reduced data colored by date
def visualize_dates(grouped_df, X, Y, xlab, ylab, title, save_title, color_choice, centers, legend):   

    #Divide data into groups (until 2019; after 2019)
    grouped_df.loc[:, 'years'] = grouped_df['years'].astype(int)
    before_covid = grouped_df[grouped_df['years'] <= 2019]
    after_covid = grouped_df[grouped_df['years'] > 2019]

    #find counts in each group
    all_group = grouped_df.groupby(['years', 'quarter'])
    before_group = before_covid.groupby(['years', 'quarter'])
    after_group = after_covid.groupby(['years', 'quarter'])

    n_colors_per_category = len(all_group)
    n_colors_BC = len(before_group)
    n_colors_AC = len(after_group)
    
    data = {'X': X,'Y': Y,'group': list(zip(grouped_df['years'], grouped_df['quarter']))}
    df1 = pd.DataFrame(data)
    grouped_df1 = df1.groupby(['group'])
    
    palette = sns.color_palette("Spectral", n_colors=n_colors_per_category)
    
    darken_factor = 0.2  # Adjust the darken factor
    adjusted_palette = []
    for color in palette:
        h, l, s = colorsys.rgb_to_hls(*color)
        adjusted_lightness = max(0.0, min(1.0, l - darken_factor))
        adjusted_rgb = colorsys.hls_to_rgb(h, adjusted_lightness, s)
        adjusted_palette.append(adjusted_rgb)


    palette = adjusted_palette[::-1]

    before_palette = palette[:n_colors_BC]
    after_palette = palette[n_colors_BC:]
    
    pink_hue = 0.84

    # Adjust the colors in after_palette to be pink shades
    for i in range(len(after_palette)):
        # Convert RGB to HLS
        r, g, b = after_palette[i]
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        
        # Adjust hue to be in the pink range
        adjusted_rgb = colorsys.hls_to_rgb(pink_hue, l, s)
    
        # Replace the original color with the adjusted one
        after_palette[i] = adjusted_rgb

    #color selection based on data group
    if color_choice == "grouped_df":
        custom_colors = before_palette + after_palette
    elif color_choice == "grouped_df_bc":
        custom_colors = before_palette
    elif color_choice == "grouped_df_ac":
        custom_colors = after_palette
    elif color_choice == "transform":
        custom_colors = after_palette

    if legend:
        legend_handles = []
        year_labels = []
    
    plt.figure(figsize=(8, 6))

    # Create scatter plot and store year labels and colors
    for (name, group), color in zip(grouped_df1, custom_colors):
        year = name[0][0]  # Extract only the year from the tuple
        if legend:
            plt.scatter(group['X'], group['Y'], label=year, s=17, color=color)
            year_labels.append(year)
        else:
            plt.scatter(group['X'], group['Y'], label=year, s=17, color=color)

    # Remove duplicates in year_labels to avoid redundant entries in the color bar
    if legend:
        unique_year_labels = list(dict.fromkeys(year_labels))

    # Plot cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', label='Cluster Centers')

    # Set plot title and labels
    #plt.title(title, fontsize=15, fontname='serif')
    #plt.xlabel(xlab, fontsize=14, fontname='serif')

    plt.xlabel(xlab, fontsize=14)
    plt.ylabel(ylab, fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    if legend:
        # Create color bar based on custom_colors and year labels
        cmap = mpl.colors.ListedColormap(custom_colors)

        # Normalize the color bar to the number of unique years
        norm = mpl.colors.BoundaryNorm(range(len(unique_year_labels) + 1), cmap.N)

        # Create the color bar and set the ticks and labels
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required for ScalarMappable

        # Adjust the ticks to be evenly spaced based on the number of unique years
        cbar = plt.colorbar(sm, ticks=range(len(unique_year_labels)))
        cbar.set_ticklabels(unique_year_labels)  # Set custom year labels for color bar

    # Save the plot
    plt.savefig(save_title, bbox_inches='tight')
    
