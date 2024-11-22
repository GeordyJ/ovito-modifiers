# Geordy Jomon

import numpy as np
import ovito
from traits.api import Float, Int, ListFloat, ListInt


# INFO: These modifiers were intended for a single trajectory frame.
# Test the modifiers are working before using them.


class AddParticleIdentifierModifier(ovito.pipeline.ModifierInterface):
    """Add particle identifiers if not present"""

    def modify(self, data: ovito.data.DataCollection, **kwargs):
        if data.particles.identifiers is None:
            data.particles_.create_property(
                'Particle Identifier',
                data=np.arange(1, len(data.particles.positions) + 1),
            )


class AlignParticlesToXAxisModifier(ovito.pipeline.ModifierInterface):
    """
    For given two particles, A and B, referenced by their pid. Orient
    the system such that particle A is at the origin and particle B is
    on the x-axis, such that A(0,0,0) and B(L,0,0), where L is the
    distance between A and B.

    pids- particle identifiers. If the pid is not specified in the input
    trajectory then it is set to the index of the positions starting at 1
    """

    pids = ListInt([0, 0], label='particle_ids')   # A id, B id

    def modify(self, data: ovito.data.DataCollection, **kwargs):
        for id in self.pids:
            if id <= 0 or id > len(data.particles.positions):
                raise ValueError('Enter valid particle_ids')
            if data.particles.identifiers is None:
                raise ValueError('Particle Identifiers Not Found')

        positions = data.particles_.positions_

        # Translate system so A is at the origin
        positions -= positions[self.pids[0] - 1]

        # Rotate AB around x axis
        vec_AB = positions[self.pids[1] - 1] - positions[self.pids[0] - 1]
        theta = np.arctan2(vec_AB[2], vec_AB[1])
        x_axis_rotation = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )
        positions = np.dot(positions, x_axis_rotation)

        # Rotate AB around z axis
        vec_AB = positions[self.pids[1] - 1] - positions[self.pids[0] - 1]
        theta = np.arctan2(vec_AB[1], vec_AB[0])
        z_axis_rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        positions = np.dot(positions, z_axis_rotation)

        data.particles_.positions_[:] = positions

        bond_AB = np.linalg.norm(
            positions[self.pids[1] - 1] - positions[self.pids[0] - 1]
        )
        # print('L_AB:', bond_AB)
        # print('Position A:', positions[self.pids[0] - 1])
        # print('Position B:', positions[self.pids[1] - 1])
        if False in np.isclose(positions[self.pids[1] - 1], [bond_AB, 0, 0]):
            raise RuntimeError('Unable to align... Check values.')


class ChangeAtomPositionModifier(ovito.pipeline.ModifierInterface):
    """
    Changes the positon of a SINGLE particle with new (x,y,z) coordinates.
    If one or more coordinate is np.NaN or numpy.NaN, then it is not
    modified.

    It only changes a SINGLE atom, cation must be taken not to cause
    overlaps or unphysical geometry.
    """

    pid = Int(0, label='particle_id')   # B id
    new_coords = ListFloat([np.NaN, np.NaN, np.NaN], label='new_position')

    def modify(self, data: ovito.data.DataCollection, **kwargs):
        if self.pid <= 0 or self.pid > len(data.particles.positions):
            raise ValueError('Enter valid particle_ids')
        if data.particles.identifiers is None:
            raise ValueError('Particle Identifiers Not Found')
        coords = data.particles_.positions_[self.pid - 1]
        for index, coord in enumerate(self.new_coords):
            if not np.isnan(coord):
                coords[index] = coord
                print(coord)

        data.particles_.positions_[self.pid - 1] = coords


class CalculateBondLengthModifier(ovito.pipeline.ModifierInterface):
    """
    Calculate bond length by first creating bonds using the builtin
    function and then calculating the bond lengths.
    """

    def modify(self, data: ovito.data.DataCollection, **kwargs):
        data.apply(
            ovito.modifiers.CreateBondsModifier(
                mode=ovito.modifiers.CreateBondsModifier.Mode.VdWRadius
            )
        )
        lengths = []
        for bond in data.particles.bonds['Topology'][...]:
            lengths.append(
                np.linalg.norm(
                    data.particles.positions[bond[1]]
                    - data.particles.positions[bond[0]]
                )
            )
        data.particles_.bonds_.create_property('Length', data=lengths)


class ChangeBondLengthModifier(ovito.pipeline.ModifierInterface):
    """
    This modifier changes the bond length of two particles A,B which
    are represented by a unique topology identifier (the first particle
    in the trajectory file is zero and so on). It moves particle B and
    all the particles bonded to B while maintaining bond angles.

    Parameters:
    bids: The unique particle identifier, obtained from ovito bond Topology.
    length: The new length of the bond.

    Note:
    If bonds are not present, it creates bonds using ovito (with VDW radius)
    """

    bids = ListInt([0, 0], label='particle_ids')   # A id, B id
    length = Float(0, label='length')

    def modify(self, data: ovito.data.DataCollection, **kwargs):
        if self.length <= 0:
            raise ValueError('The "length" parameter must be > 0')
        if self.bids[0] == self.bids[1]:
            raise ValueError(
                'The particle identifier for "bids" must not be the same'
            )
        if data.particles.bonds is None:
            # print(
            #     'Warning: No bonds found... Using ovito VDWRadius to make bonds.'
            # )
            data.apply(
                ovito.modifiers.CreateBondsModifier(
                    mode=ovito.modifiers.CreateBondsModifier.Mode.VdWRadius
                )
            )
        data.apply(
            ovito.modifiers.ComputePropertyModifier(
                operate_on='bonds',
                output_property='Length',
                expressions=['BondLength'],
            )
        )
        selected_atoms = [self.bids[1]]
        index = 0
        while index < len(selected_atoms):
            chosen_atom = selected_atoms[index]
            for bond in data.particles.bonds.topology[...]:
                if chosen_atom in bond and not np.array_equal(
                    np.sort(self.bids), np.sort(bond)
                ):
                    atom_index = bond[0] if bond[0] != chosen_atom else bond[1]
                    if atom_index not in selected_atoms:
                        selected_atoms.append(atom_index)
            index += 1

        vec_AB = (
            data.particles.positions[self.bids[1]]
            - data.particles.positions[self.bids[0]]
        )
        mag_AB = np.linalg.norm(vec_AB)
        dir_vec_AB = vec_AB / mag_AB
        disp_vec = abs(self.length - mag_AB) * dir_vec_AB

        positions = data.particles_.positions_
        for atom_index in selected_atoms:
            positions[atom_index] += disp_vec

        data.particles_.positions_[:] = positions


class ShrinkWrapSimulationBoxWithDeltaModifier(
    ovito.pipeline.ModifierInterface
):
    """
    Creates a simulation cell or adjusts an existing cell to match the
    axis-aligned bounding box of the particles. This function adds a delta
    factor to the cell size to avoid having particles on the edge. Based on-
    https://www.ovito.org/manual/python/introduction/examples/modifiers/shrink_wrap_box.html
    """

    delta = Int(0, label='delta')

    def modify(self, data: ovito.data.DataCollection, **kwargs):
        if not data.particles or data.particles.count == 0:
            return

        coords_min = np.amin(data.particles.positions, axis=0) - self.delta
        coords_max = np.amax(data.particles.positions, axis=0) + self.delta

        matrix = np.empty((3, 4))
        matrix[:, :3] = np.diag(coords_max - coords_min)
        matrix[:, 3] = coords_min

        data.create_cell(matrix, (False, False, False))


class SelectRadicalGroup(ovito.pipeline.ModifierInterface):
    """
    Selects the radical group of a molecule. For two particles A and B
    selects particle B and the group attached to it, except the A branch
    Uses ovito.modifiers.CreateBondsModifier.Mode.VdWRadius to determine
    bonds if bonds not present.
    """

    pids = ListInt([0, 0], label='particle_ids')

    def modify(self, data: ovito.data.DataCollection, **kwargs):
        if data.particles.bonds is None:
            data.apply(
                ovito.modifiers.CreateBondsModifier(
                    mode=ovito.modifiers.CreateBondsModifier.Mode.VdWRadius
                )
            )
        positions = data.particles_.positions_
        selected_ids = [self.pids[1]]
        index = 0
        while index < len(selected_ids):
            chosen_id = selected_ids[index]
            for bond in data.particles.bonds.topology[...]:
                if chosen_id in bond and not np.array_equal(
                    np.sort(self.pids), np.sort(bond)
                ):
                    atom_index = bond[0] if bond[0] != chosen_id else bond[1]
                    if atom_index not in selected_ids:
                        selected_ids.append(atom_index)
            index += 1
        sel = [0] * len(positions)
        for index, id in enumerate(selected_ids):
            sel[id] = 1
            print(index, id, positions[id])
        print(sel)
        data.particles_.create_property('Selection', data=sel)
