"""Regression tests for bootstrap statistics and Brahma snapshot selection.

Run with: python -m unittest discover -s tests -p 'test_arepo_regressions.py'
"""
import importlib.util
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import cosmo_sim_tools
from cosmo_sim_tools.brahma import snapshot

# Load this module without importing unrelated optional arepo_tools helpers.
source = pathlib.Path(cosmo_sim_tools.__file__).parent / 'arepo_tools/arepo_package.py'
spec = importlib.util.spec_from_file_location('arepo_under_test', source)
arepo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arepo)


class StatisticsTests(unittest.TestCase):
    def test_mean_and_median_remain_distinct(self):
        x = np.array([0., 0., 0., 1., 1., 1.])
        y = np.array([1., 1., 10., 2., 2., 20.])
        # A fixed resample makes the regression deterministic and exposes the
        # old duplicate-definition bug without relying on random convergence.
        with patch.object(np.random, 'choice', side_effect=lambda sample, **kw: sample.copy()):
            median = arepo.make_median_with_bootstrap(x, y, 0, 1, 2, 4)
            mean = arepo.make_mean_with_bootstrap(x, y, 0, 1, 2, 4)
        np.testing.assert_allclose(median[2], [1, 2])
        np.testing.assert_allclose(mean[2], [4, 8])
        np.testing.assert_allclose(median[3], 0)

    def test_logarithmic_statistics_match_pretransformed_inputs(self):
        x = np.geomspace(1, 1000, 60)
        y = 10 * x ** 2
        calls = [
            (arepo.mean_plot, (4,)),
            (arepo.median_plot, (4,)),
            (arepo.mean_plot2, (0, 3, 4)),
            (arepo.get_median_with_IQR, (0, 3, 4, 75)),
        ]
        for func, args in calls:
            with self.subTest(function=func.__name__):
                actual = func(x, y, True, True, *args)
                expected = func(np.log10(x), np.log10(y), False, False, *args)
                for a, e in zip(actual, expected):
                    np.testing.assert_allclose(a, e)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = pathlib.Path(self.temp.name) / 'output'
        (self.base / 'snapdir_000').mkdir(parents=True)
        (self.base / 'groups_000').mkdir()
        self.lengths = np.zeros(6, dtype=np.int64)
        self.lengths[5] = 3
        self.offsets = np.zeros(6, dtype=np.int64)
        self.offsets[5] = 2
        self.chunk_offsets = np.zeros((6, 2), dtype=np.int64)
        self.chunk_offsets[5, 1] = 3
        for chunk in range(2):
            for ordered in (False, True):
                path_func = snapshot.snapPath_groupordered if ordered else snapshot.snapPath
                with h5py.File(path_func(str(self.base), 0, chunk), 'w') as f:
                    h = f.create_group('Header')
                    counts = np.zeros(6, dtype=np.uint32)
                    counts[5] = 3
                    h.attrs['NumPart_ThisFile'] = counts
                    h.attrs['NumPart_Total'] = counts * 2
                    h.attrs['NumPart_Total_HighWord'] = np.zeros(6, dtype=np.uint32)
                    ids = np.arange(chunk * 3, chunk * 3 + 3) + (100 if ordered else 0)
                    f['PartType5/ParticleIDs'] = ids
        with h5py.File(self.base / 'groups_000/groups_000.0.hdf5', 'w') as f:
            h = f.create_group('Header')
            h.attrs['FileOffsets_Group'] = [0]
            h.attrs['FileOffsets_Subhalo'] = [0]
            h.attrs['FileOffsets_Snap'] = self.chunk_offsets
            for kind in ('Group', 'Subhalo'):
                f[f'{kind}/{kind}LenType'] = [self.lengths]
                f[f'Offsets/{kind}_SnapByType'] = [self.offsets]

    def test_ordinary_loaders_read_only_requested_particles_across_chunks(self):
        for loader in (snapshot.loadHalo, snapshot.loadSubhalo):
            with self.subTest(loader=loader.__name__):
                ids = loader(str(self.base), 0, 0, 5, fields='ParticleIDs')
                np.testing.assert_array_equal(ids, [2, 3, 4])

    def test_public_offset_file_layout(self):
        catalog = self.base / 'groups_000/groups_000.0.hdf5'
        catalog.rename(catalog.with_name('fof_subhalo_tab_000.0.hdf5'))
        offsets = pathlib.Path(snapshot.offsetPath(str(self.base), 0))
        offsets.parent.mkdir(parents=True)
        with h5py.File(offsets, 'w') as f:
            f['FileOffsets/SnapByType'] = self.chunk_offsets.T
            for kind in ('Group', 'Subhalo'):
                f[f'FileOffsets/{kind}'] = [0]
                f[f'{kind}/SnapByType'] = [self.offsets]
        for loader in (snapshot.loadHalo, snapshot.loadSubhalo):
            np.testing.assert_array_equal(
                loader(str(self.base), 0, 0, 5, fields='ParticleIDs'), [2, 3, 4])

    def test_missing_metadata_raises_instead_of_loading_all_particles(self):
        with h5py.File(self.base / 'groups_000/groups_000.0.hdf5', 'a') as f:
            del f['Offsets']
        with self.assertRaisesRegex(ValueError, 'offset metadata'):
            snapshot.loadSubhalo(str(self.base), 0, 0, 5, fields='ParticleIDs')

    def test_groupordered_loader_uses_ordered_files_and_explicit_offsets(self):
        subset = dict(lenType=self.lengths, offsetType=self.offsets, snapOffsets=self.chunk_offsets)
        actual = snapshot.loadSubhalo_groupordered(
            str(self.base), 0, 0, 5, fields='ParticleIDs', subset=subset)
        np.testing.assert_array_equal(actual, [102, 103, 104])

    def test_groupordered_loader_rejects_missing_or_pair_offsets(self):
        for subset in (None, {}, dict(lenType=self.lengths, offsetType=[2, 0],
                                     snapOffsets=self.chunk_offsets)):
            with self.subTest(subset=subset), self.assertRaises(ValueError):
                snapshot.loadSubhalo_groupordered(str(self.base), 0, 0, 5, subset=subset)

    def test_offset_lookup_rejects_invalid_ids_and_types(self):
        for object_id, kind in ((-1, 'Group'), (0.5, 'Group'), (0, 'Unknown')):
            with self.subTest(id=object_id, kind=kind), self.assertRaises(ValueError):
                snapshot.getSnapOffsets(str(self.base), 0, object_id, kind)


if __name__ == '__main__':
    unittest.main()
