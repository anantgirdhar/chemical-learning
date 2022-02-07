"""Chemical Graph Theory (CGT)

This module implements some basic functionality of chemical graph theory,
including a way to represent species as graphs with node and edge attributes
and the VF2 algorithm for (sub-)graph isomorphism checking.
"""
import networkx as nx
import matplotlib.pyplot as plt

class Specie():
    def __init__(self, adj_list, name=None):
        self._parse_adj_list(adj_list)
        if name is not None:
            self.name = name

    def _parse_adj_list(self, adj_list):
        """Parse a chemical adjacency list"""
        atoms = []
        bonds = []
        if isinstance(adj_list, str):
            adj_list = adj_list.splitlines()
        for line in adj_list:
            line = line.strip()
            tokens = line.split()
            if len(tokens) == 1:
                # This is just the name that is in the file
                self.name = tokens[0]
            elif tokens[0] == 'multiplicity':
                self.multiplicity = int(tokens[1])
            elif tokens[0].isdigit():
                atom_number = int(tokens[0])
                atoms.append((atom_number, {
                    'atom': tokens[1],
                    'u': int(tokens[2][1:]),
                    'p': int(tokens[3][1:]),
                    'c': int(tokens[4][1:]),
                    }))
                for t in tokens[5:]:
                    to_atom, bond_type = t.split(',')
                    to_atom = to_atom[1:]  # remove the leading '{'
                    to_atom = int(to_atom)
                    bond_type = bond_type[:-1]  # remove the trailing '}'
                    if atom_number < to_atom:
                        # Only include bonds from the smaller number to the
                        # larger number
                        # Since bonds are, in graph theory terms, undirected
                        # edges, we don't need to include it in both directions
                        bonds.append((atom_number, to_atom, {'type': bond_type}))
        # Now create the graph
        self.G = nx.Graph()
        self.G.add_nodes_from(atoms)
        self.G.add_edges_from(bonds)

    def plot(self, block=True):
        plt.figure()
        # Define what each bond and edge should look like
        ATOMCOLORS = {'H': '#d6d1cb', 'O': 'r', 'N': 'b', 'C': 'k'}
        BONDCOLORS = {'S': 'k', 'D': 'b', 'T': 'r'}
        BONDWIDTHS = {'S': 1, 'D': 2, 'T': 3}
        BONDSTYLES = {'S': '-', 'D': '--', 'T': '-.'}
        # Create the basic layout
        pos = nx.spring_layout(self.G, seed=3113794652, iterations=100)
        # pos = nx.planar_layout(self.G)
        # Draw the atoms
        node_colors = [ATOMCOLORS[self.G.nodes[node]['atom']] for node in self.G.nodes]
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, node_size=800, alpha=0.7)
        # Draw the bonds
        edge_colors = [BONDCOLORS[self.G.edges[bond]['type']] for bond in self.G.edges]
        edge_widths = [BONDWIDTHS[self.G.edges[bond]['type']] for bond in self.G.edges]
        edge_styles = [BONDSTYLES[self.G.edges[bond]['type']] for bond in self.G.edges]
        nx.draw_networkx_edges(self.G, pos, edge_color=edge_colors, width=edge_widths, style=edge_styles)
        # Draw the labels
        # atom_labels = {node: self.G.nodes[node]['atom'] for node in self.G.nodes}
        atom_labels = {}
        for node in self.G.nodes:
            label = '$'
            atom_type = self.G.nodes[node]['atom']
            u = self.G.nodes[node]['u']
            p = self.G.nodes[node]['p']
            c = self.G.nodes[node]['c']
            if u > 0:
                label += r'_{u' + str(u) + r'}'
            label += atom_type
            if p > 0:
                label += r'_{p' + str(p) + r'}'
            if c > 0:
                label += r'^{' + str(c) + r'}'
            label += '$'
            atom_labels[node] = label
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
        return f"<Specie {self.__str__()}>"
        return self.__str__()

    def __str__(self):
        atom_counts = self.atom_counts
        # Reorder the atoms so that they appear in a somewhat specific order
        order = ['C', 'H', 'O', 'N']
        atoms = list(atom_counts.keys())
        for atom in reversed(order):
            if atom in atoms:
                atoms.remove(atom)
                atoms.insert(0, atom)
        formula = ""
        for atom in atoms:
            count = atom_counts[atom]
            if count > 1:
                formula += f"{atom}{count}"
            else:
                formula += atom
        return formula

# Subclass the GraphMatcher to check for semantic feasibility
class ChemicalVF2(nx.algorithms.isomorphism.GraphMatcher):
    def semantic_feasibility(self, n, m):
        if self.G1.nodes[n] != self.G2.nodes[m]:
            # print(f'nodes {n} and {m} mismatch')
            return False
        for n_prime, m_prime in self.core_1.items():
            if (n, n_prime) in self.G1.edges():
                if self.G1.edges()[n, n_prime] != self.G2.edges()[m, m_prime]:
                    # print(f'edge ({n}, {n_prime}) ~= ({m}, {m_prime})')
                    return False
        return True
