from cgt import Specie

class Reaction:
    """A reaction and its associated parameters"""
    def __init__(self, rxn_string, kinetics_type, kinetics, degeneracy, duplicate, elementary_high_p, species_dictionary):
        self._parse_reaction_string(rxn_string, species_dictionary)
        self.kinetics_type = kinetics_type
        self.kinetics = kinetics
        self.degeneracy = degeneracy
        self.duplicate = duplicate
        self.elementary_high_p = elementary_high_p

    def _parse_reaction_string(self, rxn_string, species_dictionary):
        """Parse a reaction string and create reactants and products"""
        tokens = rxn_string.split()
        status = "reading reactants"
        self.reactants = []
        self.products = []
        for token in tokens:
            if token.find('=') > -1:
                # Switch to products
                status = "reading products"
                continue
            elif token == '+':
                continue
            else:
                # Add this chemical specie to the appropriate list
                if status == "reading reactants":
                    self.reactants.append(species_dictionary[token])
                elif status == "reading products":
                    self.products.append(species_dictionary[token])
                else:
                    raise Exception('Something weird happened')

    def __repr__(self):
        reactants = ' + '.join([str(r) for r in self.reactants])
        products = ' + '.join([str(p) for p in self.products])
        return f'{reactants} <=> {products}'
