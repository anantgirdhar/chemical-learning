"""Chemical Graph Theory (CGT)

This module implements some basic functionality of chemical graph theory,
including a way to represent species as graphs with node and edge attributes
and the VF2 algorithm for (sub-)graph isomorphism checking.
"""
import networkx as nx
import matplotlib.pyplot as plt

class Specie():
    def __init__(self, adj_list):
        self.G = self._parse_adj_list(adj_list)

    def _parse_adj_list(self, adj_list):
        """Parse a chemical adjacency list"""
        g = {}
        if isinstance(adj_list, str):
            adj_list = adj_list.splitlines()
        for line in adj_list:
            if line.split(' ')[0].isdigit():
                line = line.strip().split()
                atom_number = int(line[0])
                g[atom_number] = {
                        'atom_type': line[1],
                        'u': int(line[2][1:]),
                        'p': int(line[3][1:]),
                        'c': int(line[4][1:]),
                        'bonds': [],
                        }
                for bonds in line[5:]:
                    to_atom, bond_type = bonds.split(',')
                    to_atom = to_atom[1:]  # remove the leading '{'
                    to_atom = int(to_atom)
                    bond_type = bond_type[:-1]  # remove the trailing '}'
                    g[atom_number]['bonds'].append((to_atom, bond_type))
        # Create the list of bonds for easy creation
        bond_list = [(from_atom, to_atom, {'type': bond_type}) for from_atom in g.keys() for to_atom, bond_type in g[from_atom]['bonds'] if from_atom < to_atom]
        # print(bond_list)
        # Now create the graph
        G = nx.Graph()
        G.add_nodes_from([(k, {'atom': g[k]['atom_type']}) for k in g.keys()])
        G.add_edges_from(bond_list)
        return G

    def plot(self, block=True):
        plt.figure()
        # Define what each bond and edge should look like
        ATOMCOLORS = {'H': '#d6d1cb', 'O': 'r', 'N': 'b', 'C': 'k'}
        BONDCOLORS = {'S': 'k', 'D': 'b', 'T': 'r'}
        BONDWIDTHS = {'S': 1, 'D': 2, 'T': 3}
        BONDSTYLES = {'S': '-', 'D': '--', 'T': '-.'}
        # Create the basic layout
        pos = nx.spring_layout(self.G, seed=3113794652)
        # Draw the atoms
        node_colors = [ATOMCOLORS[self.G.nodes[node]['atom']] for node in self.G.nodes]
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, node_size=800, alpha=0.7)
        # Draw the bonds
        edge_colors = [BONDCOLORS[self.G.edges[bond]['type']] for bond in self.G.edges]
        edge_widths = [BONDWIDTHS[self.G.edges[bond]['type']] for bond in self.G.edges]
        edge_styles = [BONDSTYLES[self.G.edges[bond]['type']] for bond in self.G.edges]
        nx.draw_networkx_edges(self.G, pos, edge_color=edge_colors, width=edge_widths, style=edge_styles)
        # Draw the labels
        atom_labels = {node: self.G.nodes[node]['atom'] for node in self.G.nodes}
        nx.draw_networkx_labels(self.G, pos, atom_labels, font_size=22, font_color='whitesmoke')
        # nx.draw_planar(self.G, with_labels=True, node_color=node_colors, edge_color=edge_colors, width=edge_widths, style=edge_styles, font_weight='bold')
        plt.show(block=block)

    @property
    def atom_counts(self):
        counts = {}
        for i in self.G.nodes:
            atom = self.G.nodes[i]['atom']
            try:
                counts[atom] += 1
            except KeyError:
                counts[atom] = 1
        return counts

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        atom_counts = self.atom_counts
        formula = ""
        if 'C' in atom_counts:
            formula += f"C{atom_counts['C']}"
        if 'H' in atom_counts:
            formula += f"H{atom_counts['H']}"
        if 'O' in atom_counts:
            formula += f"O{atom_counts['O']}"
        if 'N' in atom_counts:
            formula += f"N{atom_counts['N']}"
        for k, v in atom_counts.items():
            if k in ['C', 'H', 'O', 'N']:
                continue
            formula += f'{k}{v}'
        return f"<Specie {formula}>"

# Subclass the GraphMatcher to check for semantic feasibility
class ChemicalVF2(nx.algorithms.isomorphism.GraphMatcher):
    def semantic_feasibility(self, n, m):
        if self.G1.nodes[n] != self.G2.nodes[m]:
            return False
        for n_prime, m_prime in self.core_1.items():
            if (n, n_prime) in self.G1.edges():
                if self.G1.edges()[n, n_prime] != self.G2.edges()[m, m_prime]:
                    return False
        return True
