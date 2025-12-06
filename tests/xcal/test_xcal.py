from citkid.xcal import xcal
import pytest
import numpy as np

################################################################################
################################# get_xcal_ix #################################
################################################################################
# Need to add empty list
ffine = np.arange(0, 100, 1, dtype = np.float64)
tfine = ffine.copy()
tnoise = np.random.permutation(np.linspace(40, 60, 100))
tnoise_glitch = np.concatenate([tnoise, [1e3]])
m = "ffine,tfine,tnoise,ix0_offset,ix1_offset,std_cutoff,ix_exp"
ix = np.random.permutation(np.arange(100))
tfine1 = tfine.copy() 
tfine1[10], tfine1[21] = tfine1[21], tfine1[10]
ffine2, tfine2 = np.flip(ffine), np.flip(tfine)
@pytest.mark.parametrize(m, [
    (ffine, tfine, [20.5], 1, 1, None, np.arange(19, 23, 1, dtype = np.int32)),
    (ffine, tfine, [20.5], 0, 0, None, np.arange(20, 22, 1, dtype = np.int32)),
    (ffine, tfine, [20.5], 1, -100, None, np.array([], dtype = np.int32)),
    (ffine, tfine, [20.5], -100, 0, None, np.array([], dtype = np.int32)),
    (ffine, tfine, [20.5], 100, 1, None, np.arange(0, 23, 1, dtype = np.int32)),
    (ffine, tfine, [20.5], 1, 100, None, np.arange(19, 100, dtype = np.int32)),
    (ffine, tfine, [20.5, 21.5], 0, 0, None, np.arange(20, 23, 1, 
                                                       dtype = np.int32)),
    (ffine, tfine, [20, 22], 0, 0, None, np.arange(19, 24, 1, 
                                                       dtype = np.int32)),
    (ffine, tfine, tnoise, 0, 0, None, np.arange(39, 62, 1, dtype = np.int32)),
    (ffine, tfine, tnoise_glitch, 0, 0, 3, np.arange(39, 62, 1, 
                                                     dtype = np.int32)),
    (ffine, tfine, tnoise_glitch, 0, 0, 11, np.arange(39, 100, 1, 
                                                      dtype = np.int32)),
    (ffine[ix], tfine[ix], [20.5], 1, 1, None, np.arange(19, 23, 1, dtype = np.int32)),
    (ffine, tfine1, [20.5], 0, 0, None, np.arange(9, 23, 1, dtype = np.int32)),
    (ffine2, tfine2, [20.5], 1, 1, None, np.arange(19, 23, 1, dtype = np.int32)),
    (ffine, tfine2, [20.5], 1, 1, None, np.arange(77, 81, 1, dtype = np.int32)),
    (ffine, tfine2, tnoise, 0, 0, None, np.arange(38, 61, 1, dtype = np.int32)),
    (ffine, tfine2, tnoise_glitch, 0, 0, None, np.arange(0, 61, 1, dtype = np.int32)),
    (ffine, tfine2, tnoise_glitch, 0, 0, 3, np.arange(38, 61, 1, dtype = np.int32)),
    (ffine, tfine2, tnoise_glitch, 0, 0, 11, np.arange(0, 61, 1, dtype = np.int32)),
    (ffine, tfine1, tnoise, 0, 0, None, np.arange(39, 62, 1, dtype = np.int32)),
    ([], [], [2.5], 0, 0, None, np.array([], dtype = np.int32)),
])
def test_get_xcal_idx(ffine, tfine, tnoise, ix0_offset, ix1_offset, 
                      std_cutoff, ix_exp):
    ix = xcal.get_xcal_idx(ffine, tfine, tnoise, 
                           ix0_offset, ix1_offset, std_cutoff)
    assert isinstance(ix, np.ndarray)
    assert ix.dtype == np.int32
    assert np.allclose(ix, ix_exp)
    

@pytest.mark.parametrize("ffine,tfine,tnoise,ix0_offset,ix1_offset,std_cutoff", [
    (['a', 1, 2], [1, 2, 3], [2], 1, 1, None),  # non-numeric ffine
    ([1, 2, 3], ['a', 2, 3], [2], 1, 1, None),  # non-numeric tfine
    ([1, 2, 3], [1, 2, 3], ['a', 2], 1, 1, None),  # non-numeric tnoise
    ([1, 2, 3], [1, 2], [2], 1, 1, None),  # mismatched ffine and tfine lengths
    ([1, 2, 3], [1, 2, 3], [2], 1.5, 1, None),  # non-integer ix0_offset
    ([1, 2, 3], [1, 2, 3], [2], 1, 'a', None),  # non-integer ix1_offset
    ([1, 2, 3], [1, 2, 3], [2], 1, 1, -1),  # negative std_cutoff
])  
def test_get_xcal_idx_invalid_input(ffine, tfine, tnoise, ix0_offset, 
                                    ix1_offset, std_cutoff):
    with pytest.raises(Exception):
        xcal.get_xcal_idx(ffine, tfine, tnoise, 
                          ix0_offset, ix1_offset, std_cutoff)
        

    