# the script is to shape the relationship of guests into a matrix for easy read in the algo
import numpy as np
import networkx as nx
import string
import pandas as pd


class RelationshipMap:
    def __init__(self, table_size=10, table_count=8, mapping_method='csv'):
        if mapping_method != 'csv':
            # guest name
            self.guest_list = list(string.ascii_uppercase)[:10]
            # get relationship edges, negative is good, positive is bad
            relationships_edges = {
                ('A', 'B'): -50,
                ('C', 'D'): -50,
                ('A', 'D'): 50,
                ('E', 'F'): 25,
                ('F', 'G'): -50,
                ('H', 'I'): -50,
                ('A', 'J'): 0
            }
            self.table_size = table_size
            self.table_count = table_count

        else:
            # read from csv
            relationship_df = pd.read_csv('relationship_mat.csv').reset_index()
            # loop through all indices and calculate relationship score
            relationships_edges = {}
            for i in range(len(relationship_df) - 1):
                for j in range(i, len(relationship_df)):
                    if j == i:
                        name_1 = relationship_df.loc[i]['First Name'] + ' ' + relationship_df.loc[i]['Last Name']
                        name_2 = relationship_df.loc[j]['First Name'] + ' ' + relationship_df.loc[j]['Last Name']
                        relationships_edges[(name_1, name_2)] = 0
                    else:
                        name_1 = relationship_df.loc[i]['First Name'] + ' ' + relationship_df.loc[i]['Last Name']
                        name_2 = relationship_df.loc[j]['First Name'] + ' ' + relationship_df.loc[j]['Last Name']
                        score_1 = int(
                            relationship_df.loc[i]['Family index'] == relationship_df.loc[j]['Family index']) * -50
                        score_2 = int(
                            relationship_df.loc[i]['Family last name'] == relationship_df.loc[j][
                                'Family last name']) * -25
                        score_3 = int(relationship_df.loc[i]['Friend (grad vs. family)'] == relationship_df.loc[j][
                            'Friend (grad vs. family)']) * -10
                        score_4 = int(relationship_df.loc[i]['Age'] == relationship_df.loc[j]['Age']) * -5
                        score = score_1 + score_2 + score_3 + score_4
                        relationships_edges[(name_1, name_2)] = score

                # update people who should not sit together or in the same table
                relationships_edges['Chuyuan Fu', 'Peter Zhao'] = 100
                relationships_edges['Kathleen McCarthy', 'Patrick McCarthy'] = 100
                relationships_edges['Evijola Llabani', 'Nicole Weygandt'] = 100

        # broadcast into a matrix
        temp_graph = nx.Graph()
        for k, v in relationships_edges.items():
            temp_graph.add_edge(k[0], k[1], weight=v)
        relationships_mat_raw = nx.to_numpy_matrix(temp_graph.to_undirected(), nodelist=temp_graph.nodes)

        # normalization
        self.relationships_mat = relationships_mat_raw / 100

        # guest name
        self.guest_list = list(temp_graph.nodes)
        self.table_size = table_size
        self.table_count = table_count
